# infer_with_support.py
import cv2, torch, numpy as np
from pathlib import Path

from train_physics_guide import (
    get_model,            # 与训练完全一致：mit_b4 + ema_momentum
    load_support_and_set  # 已在训练脚本里修复形状/归一化
)

# 仅推理用到的验证变换（避免实例化训练增强引发 warning）
def build_val_transform(img_size):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_REFLECT_101),
        ToTensorV2(),
    ])

def preprocess_keep_ratio_pad(img_rgb, img_size):
    h0, w0 = img_rgb.shape[:2]
    scale = min(img_size / h0, img_size / w0)
    new_w = int(round(w0 * scale)); new_h = int(round(h0 * scale))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top  = (img_size - new_h) // 2
    pad_left = (img_size - new_w) // 2
    pad_bottom = img_size - new_h - pad_top
    pad_right  = img_size - new_w - pad_left
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                borderType=cv2.BORDER_REFLECT_101)
    x = torch.from_numpy(padded.transpose(2, 0, 1))  # [3,H,W] uint8
    meta = dict(top=pad_top, left=pad_left, new_h=new_h, new_w=new_w,
                orig_h=h0, orig_w=w0, size=img_size)
    return x, meta

def postprocess_unpad_resize(prob_pad, meta):
    top, left = meta["top"], meta["left"]
    new_h, new_w = meta["new_h"], meta["new_w"]
    h0, w0 = meta["orig_h"], meta["orig_w"]
    prob_crop = prob_pad[top:top+new_h, left:left+new_w]
    return cv2.resize(prob_crop, (w0, h0), interpolation=cv2.INTER_LINEAR)

@torch.no_grad()
def predict_tta(model, x, scales=(0.75, 1.0, 1.25)):
    outs = []
    _, _, H, W = x.shape
    for s in scales:
        hs = int(round(H * s)); ws = int(round(W * s))
        xs = torch.nn.functional.interpolate(x, size=(hs, ws), mode="bilinear", align_corners=False)
        for hflip, vflip in [(0,0),(1,0),(0,1),(1,1)]:
            xi = xs
            if hflip: xi = torch.flip(xi, dims=[3])
            if vflip: xi = torch.flip(xi, dims=[2])
            yi = model(xi)
            if vflip: yi = torch.flip(yi, dims=[2])
            if hflip: yi = torch.flip(yi, dims=[3])
            yi = torch.nn.functional.interpolate(yi, size=(H, W), mode="bilinear", align_corners=False)
            outs.append(yi)
    return torch.stack(outs, 0).mean(0)  # [1,1,H,W]

def postprocess_morph(mask_uint8, open_iter=0, close_iter=0):
    if (open_iter <= 0) and (close_iter <= 0): return mask_uint8
    kernel = np.ones((3,3), np.uint8); m = mask_uint8.copy()
    if open_iter  > 0: m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel, iterations=open_iter)
    if close_iter > 0: m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    return m

@torch.no_grad()
def run_infer_with_support(
    ckpt_path, support_ids, img_dir, mask_dir, query_dir,
    out_dir="pred_with_support", img_size=512, thr=0.5,
    use_tta=True, tta_scales=(0.75,1.0,1.25),
    morph_open=0, morph_close=0, device="cuda"
):
    model = get_model().to(device)
    model.eval()

    # 懒初始化 1x 前向
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)
    _ = model(dummy)

    # 宽松加载权重
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if isinstance(sd.get("running_proto", None), torch.Tensor):
        try: model.running_proto = sd["running_proto"].to(device)
        except: pass
    print(f"[load] missing={missing}, unexpected={unexpected}")

    # 写入 support 原型（只创建 val_tf）
    val_tf = build_val_transform(img_size)
    load_support_and_set(model, support_ids, img_dir, mask_dir, val_tf, device=device)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    exts = (".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff")

    for p in sorted(Path(query_dir).glob("*")):
        if p.suffix.lower() not in exts: continue

        img_bgr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            print(f"[WARN] read fail: {p}"); continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr.ndim==3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)

        x_chw, meta = preprocess_keep_ratio_pad(img_rgb, img_size)
        x = (x_chw.float() / 255.0).unsqueeze(0).to(device)

        logits = predict_tta(model, x, tta_scales) if use_tta else model(x)
        prob_pad = torch.sigmoid(logits)[0,0].cpu().numpy()
        prob = postprocess_unpad_resize(prob_pad, meta)

        mask = (prob >= thr).astype("uint8") * 255
        if morph_open>0 or morph_close>0:
            mask = postprocess_morph(mask, morph_open, morph_close)

        cv2.imwrite(str(Path(out_dir)/f"{p.stem}.png"), mask)

    print(f"[DONE] Saved to: {out_dir}")

if __name__ == "__main__":
    root = Path(".")
    img_dir  = root / "images"
    mask_dir = root / "masks"
    query_dir = root / "images_exp"
    ckpt = "runs_fewshot/fold4_best.pth"
    support_ids = ["simulation_30", "simulation_45", "defect2_45"]

    run_infer_with_support(
        ckpt_path=str(ckpt),
        support_ids=support_ids,
        img_dir=str(img_dir),
        mask_dir=str(mask_dir),
        query_dir=str(query_dir),
        out_dir="pred_with_support",
        img_size=512,
        thr=0.5,
        use_tta=True,
        tta_scales=(0.75,1.0,1.25),
        morph_open=0, morph_close=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
