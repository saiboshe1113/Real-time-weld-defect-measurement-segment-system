import os, cv2, math, time, random, argparse, json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import torch.nn.functional as F

class ProtoGuidedUNet(nn.Module):
    def __init__(self, encoder_name="mit_b1", in_channels=3, classes=1, encoder_weights="imagenet",
                 ema_momentum=0.95):
        super().__init__()
        self.base = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None
        )
        # 以前这里访问 self.base.decoder.out_channels 会在某些 SMP 版本报错
        # 改为 lazy init：第一次 forward 时根据 dec_out.shape[1] 创建
        self.fuse_head = None
        self.num_classes = classes

        # 运行原型（EMA）
        self.register_buffer("running_proto", None)
        self.ema_m = ema_momentum

    @torch.no_grad()
    def set_support(self, x_support, y_support):
        self.eval()
        feats = self.base.encoder(x_support)
        deep = feats[-1]
        mh, mw = deep.shape[-2:]
        m = F.interpolate(y_support, size=(mh, mw), mode="nearest")
        denom = m.sum(dim=(2, 3)) + 1e-6
        num = (deep * m).sum(dim=(2, 3))
        proto_b = num / denom.clamp_min(1e-6)
        self.running_proto = proto_b.mean(dim=0)

    @torch.no_grad()
    def _update_running_proto(self, proto_vec):
        p = proto_vec.mean(dim=0)
        if self.running_proto is None:
            self.running_proto = p
        else:
            self.running_proto = self.ema_m * self.running_proto + (1 - self.ema_m) * p

    def _masked_avg_pool(self, feat, mask):
        mh, mw = feat.shape[-2:]
        m = F.interpolate(mask, size=(mh, mw), mode="nearest")
        denom = m.sum(dim=(2,3)) + 1e-6
        num = (feat * m).sum(dim=(2,3))
        return num / denom.clamp_min(1e-6)

    def _cosine_sim_map(self, feat, proto):
        B,C,Hf,Wf = feat.shape
        feat_n = F.normalize(feat, dim=1)
        if proto.dim() == 1:
            proto = proto[None, :].expand(B, -1)
        proto_n = F.normalize(proto, dim=1).view(B, C, 1, 1)
        return (feat_n * proto_n).sum(dim=1, keepdim=True)

    def _decode(self, feats):
        """兼容不同版本的 SMP UnetDecoder 调用方式。"""
        try:
            # 新版：decoder(features)
            return self.base.decoder(feats)
        except TypeError:
            # 旧版：decoder(*features)
            return self.base.decoder(*feats)

    def forward(self, x, y=None):
        feats = self.base.encoder(x)
        dec_out = self._decode(feats)          # [B,DecC,H,W]


        # —— Lazy init 融合头 —— #
        if self.fuse_head is None:
            dec_ch = dec_out.shape[1]
            self.fuse_head = nn.Conv2d(dec_ch + 1, self.num_classes, kernel_size=1).to(dec_out.device)

        deep = feats[-1]
        if (y is not None) and (y.max() > 0):
            with torch.no_grad():
                proto_b = self._masked_avg_pool(deep, y)
                self._update_running_proto(proto_b)
            sim = self._cosine_sim_map(deep, proto_b)
        elif (self.running_proto is not None):
            sim = self._cosine_sim_map(deep, self.running_proto)
        else:
            with torch.no_grad():
                proto_dummy = deep.mean(dim=(2,3))
            sim = self._cosine_sim_map(deep, proto_dummy)

        sim_up = F.interpolate(sim, size=dec_out.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([dec_out, sim_up], dim=1)
        logits = self.fuse_head(fused)
        return logits


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def dice_coef(pred, target, eps=1e-7):
    pred = (pred>0.5).float()
    inter = (pred*target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2*inter + eps)/(union + eps)
    return dice.mean().item()

def iou_score(logits, target, thr=0.5, eps=1e-7):
    pred = (torch.sigmoid(logits)>thr).float()
    inter = (pred*target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) - inter
    iou = (inter + eps)/(union + eps)
    return iou.mean().item()

class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, smooth=1.):
        super().__init__()
        self.alpha, self.gamma, self.smooth = alpha, gamma, smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        # Dice
        num = (probs*targets).sum(dim=(2,3))*2 + self.smooth
        den = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + self.smooth
        dice = 1 - (num/den).mean()
        # Focal（基于 BCE）
        bce = self.bce(logits, targets)
        pt = torch.exp(-bce)
        focal = (self.alpha * (1-pt)**self.gamma * bce).mean()
        return dice + focal

# -----------------------------
# Dataset
# -----------------------------
class SegDataset(Dataset):
    def __init__(self, ids, img_dir, mask_dir, tfm):
        self.ids = ids
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.tfm = tfm
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        k = self.ids[i]
        # 支持 jpg/png
        img_path = self._find_image_path(k)
        assert img_path is not None, f"Image not found for id={k}"
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read: {img_path}")
        if img.ndim==2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_dir/f"{k}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {self.mask_dir/f'{k}.png'}")
        mask = (mask>0).astype('uint8')
        out = self.tfm(image=img, mask=mask)
        x = out['image'].float()/255.0
        y = out['mask'].unsqueeze(0).float()
        return x, y
    def _find_image_path(self, stem):
        p1 = self.img_dir/f"{stem}.jpg"
        p2 = self.img_dir/f"{stem}.png"
        if p1.exists(): return p1
        if p2.exists(): return p2
        return None


def build_transforms(img_size=512):
    """
    方案B：多尺度训练（0.75x/1.0x/1.25x 等比缩放 + pad），
    最后统一 pad 到最大尺寸，保证 batch 内张量同形状。
    """
    # 三个训练尺度（等比缩放后用 PadIfNeeded 补成正方形）
    s1 = int(round(img_size * 0.75))
    s2 = int(round(img_size * 1.00))
    s3 = int(round(img_size * 1.25))
    max_s = max(s1, s2, s3)  # 统一对齐的最大边

    def scale_pad_block(size):
        # 等比缩放（最长边=size）+ pad 到 size×size
        return A.Compose([
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size,
                          border_mode=cv2.BORDER_REFLECT_101),
        ])

    # ========= 训练增强 =========
    train_tf = A.Compose([
        # 几何增强（保持比例，只做平移/等比微缩放/旋转）
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,     # 轻微等比缩放（不会拉伸）
            rotate_limit=30,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7
        ),
        A.ElasticTransform(
            alpha=10, sigma=5,
            border_mode=cv2.BORDER_REFLECT_101, p=0.2
        ),

        # 多尺度：从三种尺寸中随机选择一种（等比 + pad 到对应 size）
        A.OneOf([
            scale_pad_block(s1),
            scale_pad_block(s2),
            scale_pad_block(s3),
        ], p=1.0),

        # 关键：将所有样本统一 pad 到最大尺寸 max_s×max_s
        A.PadIfNeeded(min_height=max_s, min_width=max_s,
                      border_mode=cv2.BORDER_REFLECT_101),

        ToTensorV2(),
    ])

    # ========= 验证/测试 =========
    # 为了和训练几何一致，这里只做等比缩放 + pad 到 img_size×img_size
    val_tf = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_REFLECT_101),
        ToTensorV2(),
    ])

    return train_tf, val_tf


# -----------------------------
# Training (with K-fold)
# -----------------------------
def get_model():
    return ProtoGuidedUNet(
        encoder_name="mit_b1",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        ema_momentum=0.95,   # 可调 0.8~0.99
    )

class TverskyFocalLoss(nn.Module):
    """
    Tversky (alpha,beta) 侧重惩罚 FN/FP；再叠加 Focal BCE 稳住难例。
    推荐 alpha=0.7, beta=0.3（前景易漏时）
    """
    def __init__(self, alpha=0.8, beta=0.4, gamma=2.0, focal_alpha=0.25, smooth=1.0):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.focal_alpha = focal_alpha
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum(dim=(2,3))
        fp = (probs * (1-targets)).sum(dim=(2,3))
        fn = ((1-probs) * targets).sum(dim=(2,3))
        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
        tversky_loss = 1 - tversky.mean()

        bce = self.bce(logits, targets)
        pt = torch.exp(-bce)
        focal = (self.focal_alpha * (1-pt)**self.gamma * bce).mean()

        return tversky_loss + focal



def train_one_fold(train_ids, val_ids, args, fold_idx=0, device="cuda"):
    train_tf, val_tf = build_transforms(args.img_size)
    ds_tr = SegDataset(train_ids, args.img_dir, args.mask_dir, train_tf)
    ds_va = SegDataset(val_ids,   args.img_dir, args.mask_dir, val_tf)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=len(ds_tr)>=args.batch)
    dl_va = DataLoader(ds_va, batch_size=max(1,args.batch//2), shuffle=False, num_workers=0, pin_memory=True)

    model = get_model().to(device)
    criterion = TverskyFocalLoss(alpha=0.8, beta=0.4, gamma=2.0, focal_alpha=0.25)

    # 冻结 encoder，先训解码器
    # 冻结 encoder，先训解码器（注意：在 ProtoGuidedUNet 里是 base.encoder）
    for p in model.base.encoder.parameters():
        p.requires_grad = False

    # 建议加权重衰减，少样本更稳
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.startswith("cuda"))
    # 训练调度
    total_epochs = args.epochs
    freeze_epochs = min( max(8, total_epochs//3), total_epochs-1 )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=12, verbose=True, min_lr=1e-6
    )

    best_iou, best_path = 0.0, Path(args.out_dir)/f"fold{fold_idx}_best.pth"
    patience, bad = (20, 0)

    for ep in range(1, total_epochs+1):
        model.train()
        tr_loss = 0.0
        pbar = tqdm(dl_tr, desc=f"[Fold {fold_idx}] Epoch {ep}/{total_epochs} (train)")
        for x,y in pbar:
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.startswith("cuda")):
                logits = model(x, y)   # 训练时传入 y，用 GT 计算当批原型 & 累计 EMA
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item()*x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 验证
        model.eval()
        va_loss, va_iou, va_dice, n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for x,y in dl_va:
                x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)   # 训练时传入 y，用 GT 计算当批原型 & 累计 EMA
                loss = criterion(logits, y)
                va_loss += loss.item()*x.size(0)
                va_iou  += iou_score(logits, y)*x.size(0)
                probs = torch.sigmoid(logits)
                va_dice += dice_coef(probs, y)*x.size(0)
                n += x.size(0)
        tr_loss /= len(ds_tr) if len(ds_tr)>0 else 1
        va_loss /= n if n>0 else 1
        va_iou  /= n if n>0 else 0
        va_dice /= n if n>0 else 0

        print(f"[Fold {fold_idx}] Ep {ep}: train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  IoU={va_iou:.4f}  Dice={va_dice:.4f}")

        # 解冻 encoder
        if ep == freeze_epochs:
            for p in model.base.encoder.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min', factor=0.5, patience=12, verbose=True, min_lr=1e-6
            )
            scheduler.step(va_loss)

        if ep > freeze_epochs:
            scheduler.step(va_loss)

        # 早停与保存
        if va_iou > best_iou:
            best_iou = va_iou
            bad = 0
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"[Fold {fold_idx}] Early stopping at epoch {ep}. Best IoU={best_iou:.4f}")
                break



    return best_iou, str(best_path)

# -----------------------------
# K-fold split
# -----------------------------
def make_kfold_ids(all_ids, k=5, seed=42):
    # 少样本：尽量让每折都有缺陷样本。这里简单均匀切分（你已保证每张都有mask；若部分无缺陷，可按占比分层扩展）
    random.Random(seed).shuffle(all_ids)
    folds = []
    for i in range(k):
        val = all_ids[i::k]
        train = [x for x in all_ids if x not in val]
        folds.append((train, val))
    return folds

# -----------------------------
# Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='.', help='根目录，包含 images/ 和 masks/')
    ap.add_argument('--img_dir',  type=str, default=None, help='可指定 images 路径')
    ap.add_argument('--mask_dir', type=str, default=None, help='可指定 masks 路径')
    ap.add_argument('--out_dir',  type=str, default='runs_fewshot', help='输出模型与日志目录')
    ap.add_argument('--img_size', type=int, default=512)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--amp', action='store_true', help='开启混合精度（建议在GPU上开启）')
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    args.img_dir  = args.img_dir  or str(data_dir/'images')
    args.mask_dir = args.mask_dir or str(data_dir/'masks')
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # 收集 ID（以 images 下文件名为准，且 masks 下需有同名 .png）
    ids = []
    for p in Path(args.img_dir).glob('*'):
        if p.suffix.lower() in ('.jpg', '.png'):
            stem = p.stem
            if (Path(args.mask_dir)/f"{stem}.png").exists():
                ids.append(stem)
    ids = sorted(list(set(ids)))
    assert len(ids) >= 3, f"样本太少（{len(ids)}），至少需要 3 个样本。"

    print(f"Found {len(ids)} samples.")
    folds = min(args.folds, max(2, len(ids)//3))  # 样本仅15，折数不要太高
    kfolds = make_kfold_ids(ids, k=folds, seed=args.seed)

    summary = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for fi,(tr,va) in enumerate(kfolds, 1):
        best_iou, best_path = train_one_fold(tr, va, args, fold_idx=fi, device=device)
        summary.append({"fold": fi, "val_iou": best_iou, "best": best_path})

    with open(Path(args.out_dir)/"summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("=== K-fold summary ===")
    for r in summary:
        print(r)

def load_support_and_set(model, support_ids, img_dir, mask_dir, tfm, device="cuda"):
    xs, ys = [], []
    for sid in support_ids:
        # 找图像
        p_img = None
        for ext in (".jpg",".png",".jpeg",".bmp",".tif",".tiff"):
            q = Path(img_dir) / f"{sid}{ext}"
            if q.exists():
                p_img = str(q); break
        assert p_img is not None, f"Support image not found for id={sid}"

        img = cv2.imread(p_img, cv2.IMREAD_UNCHANGED)
        if img.ndim==2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:           img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 找掩膜
        p_mask = Path(mask_dir) / f"{sid}.png"
        assert p_mask.exists(), f"Support mask not found: {p_mask}"
        mask = cv2.imread(str(p_mask), cv2.IMREAD_GRAYSCALE)
        mask = (mask>0).astype("uint8")

        out = tfm(image=img, mask=mask)  # 要求 tfm 含 ToTensorV2()

        xi = out['image']        # [3,H,W] (torch.uint8/float)
        mi = out['mask']         # [H,W]   (torch)

        # === 关键：加 batch 维，并做归一化到 [0,1] ===
        xi = (xi.float() / 255.0).unsqueeze(0)        # -> [1,3,H,W]
        mi = mi.unsqueeze(0).unsqueeze(0).float()     # -> [1,1,H,W]

        xs.append(xi); ys.append(mi)

    x_sup = torch.cat(xs, dim=0).to(device)   # [B,3,H,W]
    y_sup = torch.cat(ys, dim=0).to(device)   # [B,1,H,W]

    # 可选：简单的形状检查，首次跑时开一下
    # assert x_sup.ndim == 4 and y_sup.ndim == 4, f"bad shape: {x_sup.shape}, {y_sup.shape}"

    model.set_support(x_sup, y_sup)
    print(f"[Support] set with {len(support_ids)} samples.")


if __name__ == "__main__":
    main()
