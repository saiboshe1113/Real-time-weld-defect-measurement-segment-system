# mini_gui_infer_with_support.py
# 本地桌面小程序：Few-shot 支持集 + 分割推理（单图预览 & 批量处理）
# 依赖：pip install pillow albumentations opencv-python numpy torch torchvision

import os, io, time, zipfile, threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2, torch, numpy as np
from PIL import Image, ImageTk

# === 引入你已有的训练脚本里的接口 ===
from train_physics_guide import (
    get_model,            # 与训练完全一致：mit_b4 + ema_momentum
    load_support_and_set  # 已在训练脚本里修复形状/归一化
)

# -------- 与原推理脚本一致/轻改的工具函数 --------
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
    if open_iter  > 0: m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel, iterations=int(open_iter))
    if close_iter > 0: m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=int(close_iter))
    return m

def make_overlay(img_rgb, mask_u8, alpha=0.45):
    img = img_rgb.astype(np.float32)
    color = np.zeros_like(img, dtype=np.float32)
    color[..., 0] = 255  # 红色
    m3 = (mask_u8 > 0)[..., None].astype(np.float32)
    out = img * (1 - alpha*m3) + color * (alpha*m3)
    return out.clip(0,255).astype(np.uint8)

def parse_support_ids(s: str):
    if not s: return []
    parts = [p.strip() for p in s.replace("\n", ",").replace(" ", ",").split(",")]
    return [p for p in parts if p]

def parse_scales(s: str):
    if not s: return (0.75, 1.0, 1.25)
    try:
        vals = [float(x.strip()) for x in s.split(",") if x.strip()]
        return tuple(vals) if len(vals)>0 else (0.75, 1.0, 1.25)
    except:
        return (0.75, 1.0, 1.25)

# -------- 推理会话封装 --------
class InferenceSession:
    def __init__(self, ckpt_path, support_ids, img_dir, mask_dir, img_size=512, device="cuda"):
        self.device = device
        self.img_size = int(img_size)
        self.support_ids = support_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.model = get_model().to(device).eval()
        _ = self.model(torch.zeros(1,3,self.img_size,self.img_size, device=device))

        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("model", ckpt)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if isinstance(sd.get("running_proto", None), torch.Tensor):
            try: self.model.running_proto = sd["running_proto"].to(device)
            except: pass

        self.val_tf = build_val_transform(self.img_size)
        load_support_and_set(self.model, support_ids, img_dir, mask_dir, self.val_tf, device=device)

        self.load_log = f"weights loaded (missing={len(missing)}, unexpected={len(unexpected)})"

    @torch.no_grad()
    def infer_rgb(self, img_rgb, thr=0.5, use_tta=True, tta_scales=(0.75,1.0,1.25), morph_open=0, morph_close=0):
        x_chw, meta = preprocess_keep_ratio_pad(img_rgb, self.img_size)
        x = (x_chw.float() / 255.0).unsqueeze(0).to(self.device)
        logits = predict_tta(self.model, x, tta_scales) if use_tta else self.model(x)
        prob_pad = torch.sigmoid(logits)[0,0].cpu().numpy()
        prob = postprocess_unpad_resize(prob_pad, meta)
        mask = (prob >= float(thr)).astype("uint8") * 255
        if (morph_open>0) or (morph_close>0):
            mask = postprocess_morph(mask, morph_open, morph_close)
        return prob, mask

# -------- Tkinter GUI --------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Few-shot 分割小程序（Support + 推理）")
        root.geometry("1180x720")

        # 状态
        self.sess = None
        self.current_img_rgb = None
        self.current_img_path = None
        self.photo_left = None
        self.photo_right = None

        # 顶部参数区
        frm = ttk.LabelFrame(root, text="模型与支持集")
        frm.pack(side="top", fill="x", padx=8, pady=6)

        self.ckpt_var = tk.StringVar(value="runs_fewshot/fold4_best.pth")
        self.imgdir_var = tk.StringVar(value="images")
        self.maskdir_var = tk.StringVar(value="masks")
        self.support_var = tk.StringVar(value="simulation_30, simulation_45, defect2_45")
        self.imgsize_var = tk.IntVar(value=512)
        self.device_var = tk.StringVar(value="auto")

        ttk.Label(frm, text="ckpt:").grid(row=0, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.ckpt_var, width=50).grid(row=0, column=1, sticky="we", padx=3)
        ttk.Button(frm, text="浏览", command=self.browse_ckpt).grid(row=0, column=2, padx=3)

        ttk.Label(frm, text="images:").grid(row=0, column=3, sticky="e")
        ttk.Entry(frm, textvariable=self.imgdir_var, width=28).grid(row=0, column=4, sticky="we", padx=3)
        ttk.Button(frm, text="浏览", command=self.browse_imgdir).grid(row=0, column=5, padx=3)

        ttk.Label(frm, text="masks:").grid(row=0, column=6, sticky="e")
        ttk.Entry(frm, textvariable=self.maskdir_var, width=28).grid(row=0, column=7, sticky="we", padx=3)
        ttk.Button(frm, text="浏览", command=self.browse_maskdir).grid(row=0, column=8, padx=3)

        ttk.Label(frm, text="support IDs:").grid(row=1, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.support_var, width=50).grid(row=1, column=1, columnspan=2, sticky="we", padx=3)

        ttk.Label(frm, text="img_size:").grid(row=1, column=3, sticky="e")
        ttk.Spinbox(frm, from_=256, to=1024, increment=32, textvariable=self.imgsize_var, width=7).grid(row=1, column=4, sticky="w")

        ttk.Label(frm, text="device:").grid(row=1, column=6, sticky="e")
        ttk.Combobox(frm, values=["auto","cuda","cpu"], textvariable=self.device_var, width=10, state="readonly").grid(row=1, column=7, sticky="w")

        self.btn_load = ttk.Button(frm, text="加载模型 & Support", command=self.thread_load)
        self.btn_load.grid(row=0, column=9, rowspan=2, padx=6)

        frm.grid_columnconfigure(1, weight=1)
        frm.grid_columnconfigure(4, weight=0)
        frm.grid_columnconfigure(7, weight=0)

        # 推理参数
        pfrm = ttk.LabelFrame(root, text="推理参数")
        pfrm.pack(side="top", fill="x", padx=8, pady=6)

        self.thr_var = tk.DoubleVar(value=0.5)
        self.tta_var = tk.BooleanVar(value=True)
        self.scales_var = tk.StringVar(value="0.75,1.0,1.25")
        self.open_var = tk.IntVar(value=0)
        self.close_var = tk.IntVar(value=0)
        self.alpha_var = tk.DoubleVar(value=0.45)

        ttk.Label(pfrm, text="thr:").grid(row=0, column=0, sticky="e")
        ttk.Scale(pfrm, from_=0.0, to=1.0, variable=self.thr_var, orient="horizontal", length=180).grid(row=0, column=1, sticky="w", padx=3)
        ttk.Label(pfrm, textvariable=self.thr_var, width=6).grid(row=0, column=2, sticky="w")

        ttk.Checkbutton(pfrm, text="启用TTA", variable=self.tta_var).grid(row=0, column=3, sticky="w")
        ttk.Label(pfrm, text="scales:").grid(row=0, column=4, sticky="e")
        ttk.Entry(pfrm, textvariable=self.scales_var, width=18).grid(row=0, column=5, sticky="w", padx=3)

        ttk.Label(pfrm, text="Open:").grid(row=0, column=6, sticky="e")
        ttk.Spinbox(pfrm, from_=0, to=10, textvariable=self.open_var, width=5).grid(row=0, column=7, sticky="w")
        ttk.Label(pfrm, text="Close:").grid(row=0, column=8, sticky="e")
        ttk.Spinbox(pfrm, from_=0, to=10, textvariable=self.close_var, width=5).grid(row=0, column=9, sticky="w")

        ttk.Label(pfrm, text="Overlay透明度:").grid(row=0, column=10, sticky="e")
        ttk.Scale(pfrm, from_=0.0, to=1.0, variable=self.alpha_var, orient="horizontal", length=150).grid(row=0, column=11, sticky="w", padx=3)

        # 中部：按钮 & 预览
        ctr = ttk.Frame(root)
        ctr.pack(side="top", fill="both", expand=True, padx=8, pady=6)

        lbtns = ttk.Frame(ctr)
        lbtns.pack(side="left", fill="y", padx=(0,8))
        ttk.Button(lbtns, text="选择图片", command=self.choose_image).pack(fill="x", pady=3)
        ttk.Button(lbtns, text="推理此图", command=self.thread_predict_one).pack(fill="x", pady=3)
        ttk.Separator(lbtns, orient="horizontal").pack(fill="x", pady=6)
        ttk.Button(lbtns, text="选择文件夹(批量)", command=self.choose_folder).pack(fill="x", pady=3)
        ttk.Button(lbtns, text="批量推理并保存", command=self.thread_predict_folder).pack(fill="x", pady=3)

        # 预览区
        view = ttk.Frame(ctr)
        view.pack(side="left", fill="both", expand=True)

        self.canvas_left = tk.Label(view, text="原图/Overlay", bg="#222", fg="#ddd")
        self.canvas_left.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        self.canvas_right = tk.Label(view, text="Mask", bg="#222", fg="#ddd")
        self.canvas_right.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        # 日志
        logf = ttk.LabelFrame(root, text="日志")
        logf.pack(side="bottom", fill="x", padx=8, pady=6)
        self.log = tk.Text(logf, height=6)
        self.log.pack(fill="both", expand=True)

        self.append_log("就绪：请先加载模型与支持集。")

    # -------- GUI 事件 --------
    def browse_ckpt(self):
        p = filedialog.askopenfilename(title="选择权重文件", filetypes=[("PyTorch", "*.pth *.pt"), ("All", "*.*")])
        if p: self.ckpt_var.set(p)

    def browse_imgdir(self):
        d = filedialog.askdirectory(title="选择 images 目录")
        if d: self.imgdir_var.set(d)

    def browse_maskdir(self):
        d = filedialog.askdirectory(title="选择 masks 目录")
        if d: self.maskdir_var.set(d)

    def choose_image(self):
        p = filedialog.askopenfilename(title="选择图片", filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if not p: return
        self.current_img_path = p
        img_bgr = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            messagebox.showerror("错误", "读取图片失败")
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr.ndim==3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        self.current_img_rgb = img_rgb
        self.show_image(self.canvas_left, img_rgb, is_mask=False)
        self.canvas_right.config(image="", text="Mask")
        self.append_log(f"已加载图片：{p}")

    def choose_folder(self):
        d = filedialog.askdirectory(title="选择待推理文件夹")
        if d:
            self.query_dir = d
            self.append_log(f"批量目录：{d}")

    def thread_load(self):
        t = threading.Thread(target=self.on_load, daemon=True)
        t.start()

    def on_load(self):
        try:
            ckpt = self.ckpt_var.get()
            img_dir = self.imgdir_var.get()
            mask_dir = self.maskdir_var.get()
            sup_ids = parse_support_ids(self.support_var.get())
            if not (ckpt and sup_ids):
                messagebox.showwarning("提示", "请填写 ckpt 与 support IDs")
                return
            dev_choice = self.device_var.get()
            device = dev_choice if dev_choice in ["cpu","cuda"] else ("cuda" if torch.cuda.is_available() else "cpu")

            self.append_log("开始加载模型与支持集…")
            self.sess = InferenceSession(
                ckpt_path=ckpt,
                support_ids=sup_ids,
                img_dir=img_dir,
                mask_dir=mask_dir,
                img_size=self.imgsize_var.get(),
                device=device
            )
            self.append_log(f"✅ 加载完成：{self.sess.load_log} | device={device} | support={sup_ids}")
        except Exception as e:
            self.append_log(f"❌ 加载失败：{e}")
            messagebox.showerror("加载失败", str(e))

    def thread_predict_one(self):
        t = threading.Thread(target=self.on_predict_one, daemon=True)
        t.start()

    def on_predict_one(self):
        if self.sess is None:
            messagebox.showwarning("提示", "请先加载模型与支持集")
            return
        if self.current_img_rgb is None:
            messagebox.showwarning("提示", "请先选择图片")
            return
        try:
            thr = float(self.thr_var.get())
            use_tta = bool(self.tta_var.get())
            scales = parse_scales(self.scales_var.get())
            mopen = int(self.open_var.get())
            mclose = int(self.close_var.get())
            alpha = float(self.alpha_var.get())

            prob, mask = self.sess.infer_rgb(self.current_img_rgb, thr=thr, use_tta=use_tta,
                                             tta_scales=scales, morph_open=mopen, morph_close=mclose)
            overlay = make_overlay(self.current_img_rgb, mask, alpha=alpha)
            self.show_image(self.canvas_left, overlay, is_mask=False)
            self.show_image(self.canvas_right, mask, is_mask=True)

            # 自动保存到同目录
            if self.current_img_path:
                out_dir = Path(self.current_img_path).parent
                stem = Path(self.current_img_path).stem
                cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask)
                cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                self.append_log(f"完成：thr={thr}, TTA={use_tta}, scales={scales}, open={mopen}, close={mclose}, alpha={alpha}\n已保存到：{out_dir}")
            else:
                self.append_log("完成（未保存：无原图路径）。")
        except Exception as e:
            self.append_log(f"❌ 推理失败：{e}")
            messagebox.showerror("推理失败", str(e))

    def thread_predict_folder(self):
        t = threading.Thread(target=self.on_predict_folder, daemon=True)
        t.start()

    def on_predict_folder(self):
        if self.sess is None:
            messagebox.showwarning("提示", "请先加载模型与支持集")
            return
        query_dir = getattr(self, "query_dir", None)
        if not query_dir or not Path(query_dir).exists():
            messagebox.showwarning("提示", "请先选择批量文件夹")
            return
        try:
            out_dir = filedialog.askdirectory(title="选择保存结果的文件夹（将写入 *_mask/overlay）")
            if not out_dir: return
            out_dir = Path(out_dir) / time.strftime("%Y%m%d-%H%M%S")
            out_dir.mkdir(parents=True, exist_ok=True)

            exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")
            paths = [p for p in sorted(Path(query_dir).glob("*")) if p.suffix.lower() in exts]
            if not paths:
                self.append_log("目标文件夹没有图片。")
                return

            thr = float(self.thr_var.get())
            use_tta = bool(self.tta_var.get())
            scales = parse_scales(self.scales_var.get())
            mopen = int(self.open_var.get()); mclose = int(self.close_var.get())
            alpha = float(self.alpha_var.get())

            n_ok = 0
            for p in paths:
                img_bgr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr.ndim==3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
                prob, mask = self.sess.infer_rgb(img_rgb, thr=thr, use_tta=use_tta,
                                                 tta_scales=scales, morph_open=mopen, morph_close=mclose)
                overlay = make_overlay(img_rgb, mask, alpha=alpha)

                cv2.imwrite(str(out_dir / f"{p.stem}_mask.png"), mask)
                cv2.imwrite(str(out_dir / f"{p.stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                n_ok += 1

            self.append_log(f"批量完成：{n_ok}/{len(paths)} 张，保存至 {out_dir}")
            messagebox.showinfo("完成", f"批量完成：{n_ok} 张\n输出：{out_dir}")
        except Exception as e:
            self.append_log(f"❌ 批量失败：{e}")
            messagebox.showerror("批量失败", str(e))

    # -------- 显示与日志 --------
    def show_image(self, widget: tk.Label, arr: np.ndarray, is_mask=False):
        if is_mask:
            # 单通道转灰度/Pseudo-color都行，这里灰度
            if arr.ndim==2:
                im = Image.fromarray(arr)
            else:
                im = Image.fromarray(arr[...,0])
        else:
            im = Image.fromarray(arr)  # RGB

        # 自适应缩放
        max_w = int(self.root.winfo_width()/2) - 60
        max_h = int(self.root.winfo_height()/2) + 120
        w, h = im.size
        scale = min(max_w / max(1,w), max_h / max(1,h), 1.0)
        if scale < 1.0:
            im = im.resize((int(w*scale), int(h*scale)), Image.BILINEAR)

        photo = ImageTk.PhotoImage(im)
        widget.config(image=photo, text="")
        # 保存引用防止被GC
        if widget is self.canvas_left:
            self.photo_left = photo
        else:
            self.photo_right = photo

    def append_log(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")

def main():
    # 自动选择设备
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
