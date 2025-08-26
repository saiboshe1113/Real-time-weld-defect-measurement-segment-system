# ect_realtime_seg_with_mask.py
# Real-time ECT acquisition + reconstruction + segmentation
# + bottom history (recon & seg), one toggle controls BOTH main seg & seg-history
# Author: you + assistant
# Date: 2025-08-24

import os
os.environ.setdefault("MPLBACKEND", "Qt5Agg")

import math, socket, time
from collections import deque
import numpy as np
import cv2
import torch
from threading import Thread

import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "Qt5Agg"))
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# ===================== 用户可配置 =====================
USE_FAKE_DATA = False
CKPT_PATH = "runs_fewshot/fold4_best.pth"
IMG_SIZE = 512

# 你的训练图是 JET
INPUT_MODE = "jet"                       # "jet" or "gray"

# 分割输入归一化（与训练/重构色标一致）
SEG_INPUT_NORM = "fixed"                 # "fixed"|"global"|"roi"
FIXED_NORM_RANGE = (5e5, 5e6)
GLOBAL_NORM_PERC = (1.0, 99.0)

# 分割输出 & 可视化
SEG_HEAD = "sigmoid"                     # 单通道
FG_INDEX = 0
MASK_THR = 0.50
THR_STEP = 0.01
SMOOTH_MASK = True
MORPH_KERNEL = 3
MORPH_ITERS  = 1

# 显示与运行
USE_AMP = True
WINDOW_SIZE = 20
UPDATE_CBAR_EVERY = 10
S_CSV = "Simplified_sensitivity.csv"

# 方向对齐
ROTATE_K = 1
FLIP_LR  = False
FLIP_UD  = False

# 重构色标
RECON_CLIM_MODE  = "fixed"               # "fixed"|"auto"
RECON_CLIM_FIXED = (5e5, 1e7)
SCALE_FAKE_TO_FIXED = True

# 底部历史滚动条
HIST_LEN        = 10
FRAME_INTERVAL  = 0.5

# 一个总开关：True 概率热力；False 黑白掩膜
SHOW_PROB = False

# Support（可选）
SUPPORT_IDS = ["defect1_45","defect1_0","defect1_90"]
SUPPORT_IMG_DIR = "images"
SUPPORT_MASK_DIR = "masks"

# ===================== 扫描参数（真机模式） =====================
scan_setting = {"Freq": 40, "CicFactor": 5, "G": 1, "H": 3, "IA": 998}
multichannel_setting = {
    0: {"E": 1, "S": 2}, 1: {"E": 1, "S": 3}, 2: {"E": 1, "S": 4}, 3: {"E": 1, "S": 5},
    4: {"E": 2, "S": 3}, 5: {"E": 2, "S": 4}, 6: {"E": 2, "S": 6}, 7: {"E": 3, "S": 4},
    8: {"E": 3, "S": 5}, 9: {"E": 3, "S": 6}, 10: {"E": 3, "S": 7}, 11: {"E": 4, "S": 5},
    12: {"E": 4, "S": 6}, 13: {"E": 4, "S": 8}, 14: {"E": 5, "S": 6}, 15: {"E": 5, "S": 7},
    16: {"E": 5, "S": 8}, 17: {"E": 5, "S": 9}, 18: {"E": 6, "S": 7}, 19: {"E": 6, "S": 8},
    20: {"E": 6, "S": 10}, 21: {"E": 7, "S": 8}, 22: {"E": 7, "S": 9}, 23: {"E": 7, "S": 10},
    24: {"E": 7, "S": 11}, 25: {"E": 8, "S": 9}, 26: {"E": 8, "S": 10}, 27: {"E": 8, "S": 12},
    28: {"E": 9, "S": 10}, 29: {"E": 9, "S": 11}, 30: {"E": 9, "S": 12}, 31: {"E": 9, "S": 13},
    32: {"E": 10, "S": 11}, 33: {"E": 10, "S": 12}, 34: {"E": 10, "S": 14}, 35: {"E": 11, "S": 12},
    36: {"E": 11, "S": 13}, 37: {"E": 11, "S": 14}, 38: {"E": 11, "S": 15}, 39: {"E": 12, "S": 13},
    40: {"E": 12, "S": 14}, 41: {"E": 12, "S": 16}, 42: {"E": 13, "S": 14}, 43: {"E": 13, "S": 15},
    44: {"E": 13, "S": 16}, 45: {"E": 14, "S": 15}, 46: {"E": 14, "S": 16}, 47: {"E": 15, "S": 16},
}
n_channel = len(multichannel_setting)

# ===================== 敏感度矩阵 =====================
def load_s_map(csv_path, w=44, l=10, n_channel_expected=None):
    if os.path.exists(csv_path):
        S_map = np.loadtxt(csv_path, delimiter=',').astype(np.float32)
        m, nPixels = S_map.shape
        assert w*l == nPixels, f"S_map 列数 {nPixels} ≠ w*l {w*l}"
        if n_channel_expected is not None:
            assert m == n_channel_expected, f"S_map 行数 {m} ≠ 通道数 {n_channel_expected}"
        return S_map, w, l
    print(f"[WARN] 未找到 {csv_path}，将随机生成 S_map（仅 demo）。")
    nPixels = w * l
    m = n_channel if n_channel_expected is None else n_channel_expected
    rng = np.random.default_rng(123)
    S_map = (rng.standard_normal((m, nPixels)) * 0.01).astype(np.float32)
    return S_map, w, l

S_map, w, l = load_s_map(S_CSV, w=44, l=10, n_channel_expected=n_channel)

# ===================== 模型 =====================
from train_physics_guide import get_model, build_transforms, load_support_and_set

# ===================== 预处理/还原 =====================
def preprocess_keep_ratio_pad(img_rgb, img_size,
                              border_mode=cv2.BORDER_REFLECT_101, border_value=0):
    h0, w0 = img_rgb.shape[:2]
    scale = min(img_size / h0, img_size / w0)
    new_w = int(round(w0 * scale)); new_h = int(round(h0 * scale))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top  = (img_size - new_h) // 2
    pad_left = (img_size - new_w) // 2
    pad_bottom = img_size - new_h - pad_top
    pad_right  = img_size - new_w - pad_left
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                borderType=border_mode, value=border_value)
    x = torch.from_numpy(padded.transpose(2, 0, 1))
    meta = dict(top=pad_top, left=pad_left, new_h=new_h, new_w=new_w, orig_h=h0, orig_w=w0)
    return x, meta

def postprocess_unpad_resize(prob_pad, meta, out_h=None, out_w=None):
    H, W = prob_pad.shape[:2]
    top   = max(0, min(int(meta.get("top", 0)), H))
    left  = max(0, min(int(meta.get("left", 0)), W))
    new_h = max(1, min(int(meta.get("new_h", H)), H - top))
    new_w = max(1, min(int(meta.get("new_w", W)), W - left))
    h0    = max(1, int(out_h if out_h is not None else meta.get("orig_h", H)))
    w0    = max(1, int(out_w if out_w is not None else meta.get("orig_w", W)))
    prob_crop = prob_pad[top:top+new_h, left:left+new_w]
    if prob_crop.size == 0: prob_crop = prob_pad
    return cv2.resize(np.ascontiguousarray(prob_crop).astype(np.float32), (w0, h0), interpolation=cv2.INTER_LINEAR)

# ===================== 方向对齐 =====================
def apply_orientation(a: np.ndarray) -> np.ndarray:
    b = a
    if ROTATE_K % 4 != 0: b = np.rot90(b, k=ROTATE_K)
    if FLIP_LR:  b = np.fliplr(b)
    if FLIP_UD:  b = np.flipud(b)
    return b

def inverse_orientation(a: np.ndarray) -> np.ndarray:
    b = a
    if FLIP_UD: b = np.flipud(b)
    if FLIP_LR: b = np.fliplr(b)
    if ROTATE_K % 4 != 0: b = np.rot90(b, k=(4-ROTATE_K)%4)
    return b

H_aligned, W_aligned = ((w, l) if (ROTATE_K % 2 == 1) else (l, w))

# ===================== UDP（真机）/ 伪数据 =====================
class UDP(Thread):
    def __init__(self):
        super().__init__()
        self.UDP_IP="192.168.1.2"; self.UDP_PORT=4592
        self.DEVICE_IP="192.168.1.10"; self.DEVICE_PORT=4590
        self.FRAME_LENGTH=32
    def init(self):
        self.local_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.local_sock.bind((self.UDP_IP, self.UDP_PORT))
        self.device_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_scan_setting(scan_setting)
        self.send_multichannel_setting(multichannel_setting)
    def write(self, message:str):
        msg = message.encode('utf-8')
        self.device_sock.sendto(msg, (self.DEVICE_IP, self.DEVICE_PORT))
        print(f"Send {msg}")
    def send_scan_setting(self, ss:dict):
        self.write(f"D{ss['Freq']}C{ss['CicFactor']}G{ss['G']}H{ss['H']}P4I{ss['IA']}S0")
    def send_multichannel_setting(self, ms:dict):
        for prefix in ("S","E"):
            self.write(prefix + "," + ",".join(str(ms[i][prefix]) for i in range(n_channel)) + ".")
    def decode_frame(self, frame:bytes):
        q = int(frame[0:8],16)/(2**math.floor(math.log(1000,2)*3))
        i = int(frame[16:24],16)/(2**math.floor(math.log(1000,2)*3))
        ch_e=int(frame[28:30],16); ch_s=int(frame[30:32],16)
        idx = next((k for k,v in multichannel_setting.items() if v['S']==ch_s and v['E']==ch_e), None)
        return i,q,idx
    def read_ECT_data(self)->np.ndarray:
        raw,_ = self.local_sock.recvfrom(10000); raw = raw[::-1]
        frames=[raw[i:i+self.FRAME_LENGTH] for i in range(0,len(raw),self.FRAME_LENGTH)][::-1]
        buf=[[] for _ in range(n_channel)]; idx_hist=[]
        for f in frames:
            i,q,idx=self.decode_frame(f); idx_hist.append(idx)
            if len(idx_hist)>3 and idx_hist[-1]==idx_hist[-4]:
                buf[idx].append((i,q)); idx_hist = idx_hist[-3:]
        length=min(len(b) for b in buf)
        for i in range(n_channel): buf[i]=buf[i][:length]
        return np.array(buf)

class FakeUDP:
    def __init__(self, S_map, l, w, n_channel, window_size=20, seed=42):
        self.S_map=S_map.astype(np.float32); self.l=l; self.w=w
        self.n_channel=n_channel; self.window_size=window_size
        self.rng=np.random.default_rng(seed)
        self.V_air=1.0+0.02*self.rng.standard_normal(self.n_channel).astype(np.float32)
        self.I_win=np.tile(self.V_air[:,None], (1,window_size)).astype(np.float32)
        self.Q_win=0.005*self.rng.standard_normal((self.n_channel,window_size)).astype(np.float32)
        self.t=0
    def init(self): pass
    def _make_phantom(self):
        y=np.linspace(0,self.l-1,self.l); x=np.linspace(0,self.w-1,self.w); X,Y=np.meshgrid(x,y)
        cx=(self.w/2)+(self.w/4)*np.sin(self.t*0.15); cy=(self.l/2)+(self.l/4)*np.cos(self.t*0.11)
        sigma=self.w*0.12; amp=0.6
        gauss = amp*np.exp(-(((X-cx)**2 + (Y-cy)**2)/(2*sigma**2)))
        phantom=np.clip(gauss + 0.02*self.rng.random((self.l,self.w)), 0, None)
        return phantom.astype(np.float32)
    def read_ECT_data(self):
        self.t += 1
        x=self._make_phantom().reshape(-1); V_diff=self.S_map @ x
        I_t=self.V_air*(1.0+V_diff).astype(np.float32)
        Q_t=0.005*self.rng.standard_normal(self.n_channel).astype(np.float32)
        self.I_win[:,:-1]=self.I_win[:,1:]; self.I_win[:,-1]=I_t
        self.Q_win[:,:-1]=self.Q_win[:,1:]; self.Q_win[:,-1]=Q_t
        return np.stack([self.I_win,self.Q_win],axis=-1)

# ===================== 参考帧 =====================
reference_data=None
last_data_buffer=None
def save_reference(event):
    global reference_data, last_data_buffer, WINDOW_SIZE
    if last_data_buffer is None:
        print("[RefButton] 无可用数据"); return
    reference_data = last_data_buffer[:, -WINDOW_SIZE:, :].copy()
    print("[RefButton] 已保存参考帧")

# ===================== 工具 =====================
def smooth_binary_mask(mask: np.ndarray, ksize=3, iters=1)->np.ndarray:
    if ksize<1 or iters<1: return mask
    if ksize%2==0: ksize += 1
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize))
    m=mask.astype(np.uint8)
    m=cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=iters)
    m=cv2.morphologyEx(m, cv2.MORPH_CLOSE,kernel, iterations=iters)
    return (m>0).astype(np.uint8)

def get_seg_norm_range(S_img_aligned: np.ndarray, S_roi: np.ndarray):
    mode=SEG_INPUT_NORM.lower()
    if mode=="fixed":
        vmin,vmax=FIXED_NORM_RANGE
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin>=vmax:
            vmin,vmax=float(np.nanmin(S_img_aligned)), float(np.nanmax(S_img_aligned))
            if vmin==vmax: vmax=vmin+1e-6
        return float(vmin), float(vmax)
    if mode=="global":
        vals=S_img_aligned[np.isfinite(S_img_aligned)].astype(np.float32)
        if vals.size==0: return 0.0,1.0
        lo,hi=np.percentile(vals, GLOBAL_NORM_PERC)
        if lo==hi: hi=lo+1e-6
        return float(lo), float(hi)
    vals=S_roi[np.isfinite(S_roi)].astype(np.float32)
    if vals.size==0: return 0.0,1.0
    lo,hi=np.percentile(vals,(1.0,99.0))
    if lo==hi: hi=lo+1e-6
    return float(lo), float(hi)

# ===================== 主程序 =====================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"): torch.backends.cudnn.benchmark=True
    autocast_dtype = torch.float16 if (device.startswith("cuda") and USE_AMP) else torch.float32
    print("[Seg] device:", device)

    seg_model = get_model().to(device).eval()
    _ = seg_model(torch.zeros(1,3,IMG_SIZE,IMG_SIZE,device=device))  # lazy head

    try:
        sd=torch.load(CKPT_PATH, map_location="cpu"); sd=sd.get("model", sd)
        missing,unexpected=seg_model.load_state_dict(sd, strict=False)
        print(f"[Seg] load: missing={len(missing)}, unexpected={len(unexpected)}")
    except Exception as e:
        print(f"[Seg][WARN] 加载权重失败：{e}（仅演示）")

    udp = FakeUDP(S_map=S_map, l=l, w=w, n_channel=n_channel, window_size=WINDOW_SIZE, seed=42) if USE_FAKE_DATA else UDP()
    udp.init()

    # ========= 画布 =========
    fig = plt.figure(figsize=(16, 6.6))
    gs  = fig.add_gridspec(2, 4, height_ratios=[9, 3], hspace=0.25, wspace=0.28)

    ax_i = fig.add_subplot(gs[0, 0])
    ax_q = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[0, 2])
    ax_s = fig.add_subplot(gs[0, 3])

    ax_hist_recon = fig.add_subplot(gs[1, 0:2])
    ax_hist_seg   = fig.add_subplot(gs[1, 2:4])
    ax_hist_recon.set_title(f"Recon history (last {HIST_LEN}, newest on right)")
    ax_hist_seg.set_title(f"Segmentation history (mask, thr={MASK_THR:.2f})")
    for ax_ in (ax_hist_recon, ax_hist_seg):
        ax_.set_xticks([]); ax_.set_yticks([])
        ax_.set_xlabel("x"); ax_.set_ylabel("y")

    I_win=np.zeros((n_channel,WINDOW_SIZE),dtype=np.float32)
    Q_win=np.zeros((n_channel,WINDOW_SIZE),dtype=np.float32)
    im=ax_i.imshow(I_win, aspect='auto', origin='upper', cmap='jet'); ax_i.set_title('I_data')
    qm=ax_q.imshow(Q_win, aspect='auto', origin='upper', cmap='jet'); ax_q.set_title('Q_data')
    cb_i=fig.colorbar(im, ax=ax_i); cb_q=fig.colorbar(qm, ax=ax_q)

    recon_im=ax_r.imshow(np.zeros((H_aligned,W_aligned),dtype=np.float32), aspect='equal', origin='lower', cmap='jet')
    cb_r=fig.colorbar(recon_im, ax=ax_r); ax_r.set_title(f"Recon (diff) [{RECON_CLIM_MODE}]")

    # 分割主图（初始按 SHOW_PROB 设置）
    seg_im=ax_s.imshow(np.zeros((H_aligned,W_aligned),dtype=np.float32 if SHOW_PROB else np.uint8),
                       origin='lower', interpolation='nearest')
    def apply_seg_view_mode():
        if SHOW_PROB:
            ax_s.set_title("Segmentation (prob)")
            seg_im.set_cmap('jet'); seg_im.set_clim(0.0, 1.0)
        else:
            ax_s.set_title(f"Segmentation (mask, thr={MASK_THR:.2f})")
            seg_im.set_cmap('gray'); seg_im.set_clim(0, 1)
    apply_seg_view_mode()
    ax_s.set_xlabel('x'); ax_s.set_ylabel('y')

    # 历史条数据与显示
    recon_hist = deque(maxlen=HIST_LEN)
    prob_hist  = deque(maxlen=HIST_LEN)
    for _ in range(HIST_LEN):
        recon_hist.append(np.zeros((H_aligned, W_aligned), dtype=np.float32))
        prob_hist.append(np.zeros((H_aligned, W_aligned), dtype=np.float32))

    hist_recon_im = ax_hist_recon.imshow(
        np.concatenate(list(recon_hist), axis=1),
        origin='lower', aspect='auto', cmap='jet'
    )
    hist_recon_im.set_clim(*RECON_CLIM_FIXED)

    hist_seg_im = ax_hist_seg.imshow(
        np.concatenate([(p >= MASK_THR).astype(np.uint8) for p in prob_hist], axis=1),
        origin='lower', aspect='auto', cmap='gray', vmin=0, vmax=1
    )
    def render_hist_seg():
        if SHOW_PROB:
            ax_hist_seg.set_title("Segmentation history (prob)")
            hist_seg_im.set_cmap('jet');  hist_seg_im.set_clim(0.0, 1.0)
            hist_seg_im.set_data(np.concatenate(list(prob_hist), axis=1))
        else:
            ax_hist_seg.set_title(f"Segmentation history (mask, thr={MASK_THR:.2f})")
            hist_seg_im.set_cmap('gray'); hist_seg_im.set_clim(0, 1)
            hist_seg_im.set_data(np.concatenate([(p >= MASK_THR).astype(np.uint8) for p in prob_hist], axis=1))

    hist_next_ts = time.monotonic()

    # ===== 按钮（抬高、保留引用） =====
    axbtn1=fig.add_axes([0.14, 0.06, 0.14, 0.05])
    axbtn2=fig.add_axes([0.32, 0.06, 0.14, 0.05])
    axbtn3=fig.add_axes([0.50, 0.06, 0.14, 0.05])
    axbtn4=fig.add_axes([0.68, 0.06, 0.14, 0.05])
    axbtn5=fig.add_axes([0.86, 0.06, 0.12, 0.05])
    for _ax in (axbtn1, axbtn2, axbtn3, axbtn4, axbtn5):
        _ax.set_zorder(10)

    btn1=Button(axbtn1, 'Save Ref')
    btn2=Button(axbtn2, 'Toggle CLim')
    def toggle_clim(event):
        global RECON_CLIM_MODE
        RECON_CLIM_MODE = "auto" if RECON_CLIM_MODE=="fixed" else "fixed"
        ax_r.set_title(f"Recon (diff) [{RECON_CLIM_MODE}]")
    btn1.on_clicked(save_reference)
    btn2.on_clicked(toggle_clim)

    # Support
    _, val_tf = build_transforms(IMG_SIZE)
    def on_load_support(event):
        if len(SUPPORT_IDS)==0:
            print("[Support] 请在脚本顶部配置 SUPPORT_IDS/IMG_DIR/MASK_DIR"); return
        try:
            load_support_and_set(seg_model, SUPPORT_IDS, SUPPORT_IMG_DIR, SUPPORT_MASK_DIR, val_tf, device=device)
        except Exception as e:
            print("[Support] 加载失败：", e)
    def on_clear_support(event):
        try:
            seg_model.running_proto = None
            print("[Support] 已清空 prototype（running_proto=None）")
        except Exception as e:
            print("[Support] 清空失败：", e)
    btn3 = Button(axbtn3, 'Load Support'); btn3.on_clicked(on_load_support)
    btn4 = Button(axbtn4, 'Clear Support'); btn4.on_clicked(on_clear_support)

    # 一个按钮切主图+历史条
    def on_toggle_segmode(event):
        global SHOW_PROB
        SHOW_PROB = not SHOW_PROB
        apply_seg_view_mode()
        render_hist_seg()
        print(f"[ui] SHOW_PROB -> {SHOW_PROB}")
    btn5 = Button(axbtn5, 'SegHist Prob'); btn5.on_clicked(on_toggle_segmode)

    fig._btns = [btn1, btn2, btn3, btn4, btn5]

    # 键盘：阈值调节（仅掩膜模式）
    def on_key(event):
        global MASK_THR
        if event.key == '[':
            MASK_THR = max(0.0, MASK_THR - THR_STEP)
        elif event.key == ']':
            MASK_THR = min(1.0, MASK_THR + THR_STEP)
        else:
            return
        if not SHOW_PROB:
            ax_s.set_title(f"Segmentation (mask, thr={MASK_THR:.2f})")
            render_hist_seg()
        print(f"[ui] MASK_THR -> {MASK_THR:.2f}")
    fig.canvas.mpl_connect('key_press_event', on_key)

    global last_data_buffer, reference_data
    frame_id=0

    while True:
        data_buffer = udp.read_ECT_data()
        last_data_buffer = data_buffer
        if data_buffer.ndim != 3:
            plt.pause(0.01); continue
        frame_id += 1

        # I/Q
        I = data_buffer[:,:,0]; Q = data_buffer[:,:,1]
        I_win[:,:] = I[:, -WINDOW_SIZE:]; Q_win[:,:] = Q[:, -WINDOW_SIZE:]

        if reference_data is not None:
            I_disp = I_win - reference_data[:,:,0]
            Q_disp = Q_win - reference_data[:,:,1]
            ax_i.set_title('I_data (diff)'); ax_q.set_title('Q_data (diff)')
        else:
            I_disp = I_win; Q_disp = Q_win
            ax_i.set_title('I_data'); ax_q.set_title('Q_data')

        im.set_data(I_disp); qm.set_data(Q_disp)
        if frame_id % UPDATE_CBAR_EVERY == 0:
            im.set_clim(I_disp.min(), I_disp.max()); cb_i.update_normal(im)
            qm.set_clim(Q_disp.min(), Q_disp.max()); cb_q.update_normal(qm)

        if reference_data is None and frame_id==10:
            reference_data = last_data_buffer.copy()
            print("[AutoRef] 参考帧已保存。")

        # 重建 + 分割
        if reference_data is not None:
            V_sel = I[:, -1]; V_air = reference_data[:, 0, 0]
            V_diff = (V_sel - V_air)
            S_rec = np.real(S_map.T.dot(V_diff))
            S_img = S_rec.reshape((l, w))
            S_img_aligned = apply_orientation(S_img)

            if USE_FAKE_DATA and SCALE_FAKE_TO_FIXED:
                a,b = float(np.nanmin(S_img_aligned)), float(np.nanmax(S_img_aligned))
                if np.isfinite(a) and np.isfinite(b) and b>a:
                    lo,hi = RECON_CLIM_FIXED
                    S_img_aligned = (S_img_aligned - a)/(b-a)*(hi-lo) + lo

            recon_im.set_data(S_img_aligned)
            if RECON_CLIM_MODE=="fixed":
                recon_im.set_clim(*RECON_CLIM_FIXED); cb_r.update_normal(recon_im)
            else:
                if frame_id % UPDATE_CBAR_EVERY == 0:
                    vmin=float(np.nanpercentile(S_img_aligned, 1.0))
                    vmax=float(np.nanpercentile(S_img_aligned, 99.0))
                    if vmin==vmax: vmax = vmin+1e-6
                    recon_im.set_clim(vmin, vmax); cb_r.update_normal(recon_im)

            # 分割输入
            S_roi = S_img_aligned
            vmin_use, vmax_use = FIXED_NORM_RANGE if SEG_INPUT_NORM=="fixed" else get_seg_norm_range(S_img_aligned, S_roi)
            den = max(1e-12, float(vmax_use - vmin_use))
            img_norm = np.clip((np.nan_to_num(S_roi.astype(np.float32)) - vmin_use)/den, 0.0, 1.0)
            if INPUT_MODE=="jet":
                img_u8 = (img_norm*255).astype(np.uint8)
                img_bgr = cv2.applyColorMap(img_u8, cv2.COLORMAP_JET)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            else:
                img_u8 = (img_norm*255).astype(np.uint8)
                img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)

            x_chw, meta = preprocess_keep_ratio_pad(img_rgb, IMG_SIZE, border_mode=cv2.BORDER_REFLECT_101)
            x = (x_chw.float()/255.0).unsqueeze(0).to(device, non_blocking=True)

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=autocast_dtype,
                                    enabled=(device.startswith("cuda") and USE_AMP)):
                    logits = seg_model(x)
                    if SEG_HEAD == "sigmoid":
                        prob_pad = torch.sigmoid(logits[:, FG_INDEX]).squeeze(0).detach().cpu().numpy()
                    else:
                        prob_pad = torch.softmax(logits, dim=1)[:, FG_INDEX].squeeze(0).detach().cpu().numpy()

            prob = postprocess_unpad_resize(prob_pad, meta, out_h=H_aligned, out_w=W_aligned)
            mask_bin = (prob >= MASK_THR).astype(np.uint8)
            if SMOOTH_MASK: mask_bin = smooth_binary_mask(mask_bin, MORPH_KERNEL, MORPH_ITERS)

            # 主图：跟随 SHOW_PROB
            if SHOW_PROB:
                seg_im.set_data(prob);     seg_im.set_cmap('jet');  seg_im.set_clim(0.0, 1.0)
            else:
                seg_im.set_data(mask_bin); seg_im.set_cmap('gray'); seg_im.set_clim(0, 1)

            # 历史条推进
            now = time.monotonic()
            if now >= hist_next_ts:
                recon_hist.append(S_img_aligned.copy())
                prob_hist.append(prob.copy())
                hist_recon_im.set_data(np.concatenate(list(recon_hist), axis=1))
                hist_recon_im.set_clim(*recon_im.get_clim())
                render_hist_seg()
                hist_next_ts = now + FRAME_INTERVAL

        fig.canvas.draw_idle()
        plt.pause(0.01)

if __name__ == "__main__":
    main()
