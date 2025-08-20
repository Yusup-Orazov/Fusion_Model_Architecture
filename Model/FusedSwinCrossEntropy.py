#!/usr/bin/env python3
"""
train_multimodal_fusion_ce.py
─────────────────────────────
Multi‑modal joint **2‑D + 3‑D** semantic‑segmentation training on the
UXB dataset (TPI images + LAS point‑clouds), organised in numbered
pair folders.

Losses
──────
• 2‑D branch  → **Class‑weighted Cross‑Entropy**  
• 3‑D branch  → **Class‑weighted Cross‑Entropy**  
  (both put heavy weight on minority classes 1 & 2)

Sampler
───────
• Minority‑focused **BalancedSampler** (≈ 70 % tiles that contain
  class 1 or 2 voxels)

Model & schedule
────────────────
• 2‑D branch ‒ SegFormer‑B0 backbone + 1×1 conv head  
• 3‑D branch ‒ MONAI Swin UNETR‑large (feature_size = 96)  
• Early‑fusion in 3‑D space  
• Two‑stage warm‑up, early stopping (patience = 10 epochs)

Outputs
───────
• Best weights → `fused_tpi_clr_swin_best.pth`  
• Inference: volumes (`pred_XXXX.npy`) and colourised LAS files
  (`pred_XXXX.las`) in `predict_out/`
"""
from pathlib import Path
import gc
import numpy as np
from PIL import Image
import laspy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

from transformers import SegformerFeatureExtractor as AutoImageProcessor
from transformers import SegformerModel

from monai.networks.nets import SwinUNETR

# ───────── CONFIG ────────────────────────────────────────────────────
ROOT            = Path("Old_data_2d+3d/uxb_tpi_clr_with_masks_paired")
GRID_SIZE       = (32, 256, 256)    # depth, height, width
NUM_CLASSES     = 3
BATCH_SIZE      = 1
EPOCHS          = 100
NUM_WORKERS     = 1
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR_WARMUP       = 2e-7
LR_MAIN         = 2e-5
WARMUP_EPOCHS   = 20
PATIENCE        = 10
BEST_PATH       = Path("FusedSwinCrossEntropy_tpi_clr_swin_best.pth")

# class weights (bias toward plazuela & structure)
CE_W_2D = torch.tensor([0.05, 0.45, 0.50])
CE_W_3D = CE_W_2D.clone()

LABEL_MAP_2D     = {0: 0, 76: 1, 150: 2}        # ground, plazuela, structure
LABEL_MAP_3D     = {2: 0, 27: 1, 6: 2}
INV_LABEL_MAP_3D = {v: k for k, v in LABEL_MAP_3D.items()}

# ───────── METRICS ──────────────────────────────────────────────────
def _cm_upd(cm, pred, gt):
    valid = (gt >= 0) & (gt < NUM_CLASSES)
    cm += torch.bincount(NUM_CLASSES * gt[valid] + pred[valid],
                         minlength=NUM_CLASSES**2).view(NUM_CLASSES, NUM_CLASSES)

def _cm_reduce(cm):
    eps   = 1e-7
    tp    = cm.diag()
    fp    = cm.sum(0) - tp
    fn    = cm.sum(1) - tp
    prec  = tp / (tp + fp + eps)
    rec   = tp / (tp + fn + eps)
    f1    = 2 * prec * rec / (prec + rec + eps)
    iou   = tp / (tp + fp + fn + eps)
    acc_c = tp / (tp + fn + eps)
    acc_g = tp.sum() / cm.sum()
    return acc_g, iou.cpu(), f1.cpu(), rec.cpu(), prec.cpu(), acc_c.cpu()

_fmt = lambda arr: ", ".join(f"{x:.3f}" for x in arr)

# ───────── DATASET ──────────────────────────────────────────────────
class UXBDataset(Dataset):
    def __init__(self, root: Path, split: str):
        self.root  = root / split
        self.pairs = sorted([d for d in self.root.iterdir()
                             if d.is_dir() and d.name.startswith("pair_")])
        self.proc  = AutoImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512", use_fast=False)

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        pdir      = self.pairs[idx]
        img_path  = next(pdir.glob("tpi_100m*.png"))
        mask_path = next(pdir.glob("uxb_msk_pl_st*.png"))
        las_path  = next(pdir.glob("*.las"))

        # 2‑D inputs
        img = Image.open(img_path).convert("RGB")
        enc = self.proc(images=img, return_tensors="pt")
        pv2 = enc["pixel_values"].squeeze(0)           # (3,512,512)

        # 3‑D inputs
        vol = self.las_to_volume(las_path)             # (32,256,256)
        pv3 = torch.from_numpy(vol[None]).float()      # (1,32,256,256)

        sample = {"pv2d": pv2, "pv3d": pv3, "las_path": str(las_path)}

        if self.root.name != "predict":
            arr  = np.array(Image.open(mask_path).convert("L"))
            lbl2 = np.zeros_like(arr)
            for src, dst in LABEL_MAP_2D.items():
                lbl2[arr == src] = dst
            lbl2 = Image.fromarray(lbl2.astype(np.uint8))
            lbl2 = lbl2.resize((GRID_SIZE[2], GRID_SIZE[1]), Image.NEAREST)
            sample.update({
                "lbl2d": torch.from_numpy(np.array(lbl2)).long(),
                "lbl3d": torch.from_numpy(vol).long()
            })
        return sample

    @staticmethod
    def las_to_volume(p: Path, grid=GRID_SIZE):
        las = laspy.read(p)
        x, y, z, cls = las.x, las.y, las.z, las.classification
        xi = ((x - x.min()) / (x.max() - x.min()) * (grid[2] - 1)).astype(int)
        yi = ((y - y.min()) / (y.max() - y.min()) * (grid[1] - 1)).astype(int)
        zi = ((z - z.min()) / (z.max() - z.min()) * (grid[0] - 1)).astype(int)
        vol = np.zeros(grid, np.int64)
        for src, dst in LABEL_MAP_3D.items():
            m = cls == src
            vol[zi[m], yi[m], xi[m]] = dst
        return vol

def collate(batch):
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch]) if torch.is_tensor(batch[0][k]) else [b[k] for b in batch]
    return out

# ───────── SAMPLER ─────────────────────────────────────────────────
class BalancedSampler(Sampler):
    def __init__(self, ds, minority_ids=(1, 2), minority_frac=0.7):
        self.ds = ds
        idx_min, idx_maj = [], []
        print("Scanning dataset for minority‑class voxels …")
        for i in range(len(ds)):
            vol = ds[i]["pv3d"][0].numpy()
            (idx_min if np.isin(vol, minority_ids).any() else idx_maj).append(i)

        if not idx_maj or not idx_min:
            self.indices, self.prob = (idx_min or idx_maj), None
        else:
            self.indices = idx_min + idx_maj
            p_min = minority_frac / len(idx_min)
            p_maj = (1 - minority_frac) / len(idx_maj)
            prob  = [p_min]*len(idx_min) + [p_maj]*len(idx_maj)
            self.prob = np.array(prob) / sum(prob)

    def __iter__(self):
        if self.prob is None:
            return iter(np.random.choice(self.indices, len(self.indices)))
        return iter(np.random.choice(self.indices, len(self.indices), p=self.prob))

    def __len__(self): return len(self.indices)

# ───────── MODEL ──────────────────────────────────────────────────
class FusedSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.back2d = SegformerModel.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            ignore_mismatched_sizes=True)
        c2 = self.back2d.config.hidden_sizes[-1]
        self.head2d = nn.Conv2d(c2, NUM_CLASSES, 1)

        self.back3d = SwinUNETR(
            img_size=GRID_SIZE,
            in_channels=1,
            out_channels=64,
            feature_size=96,
            spatial_dims=3
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(64 + c2, 128, 3, padding=1),
            nn.ReLU(True),
            nn.Conv3d(128, NUM_CLASSES, 1)
        )

    def forward(self, p2, p3):
        B, _, D, H, W = p3.shape
        f2  = self.back2d(p2).last_hidden_state      # (B,c2,16,16)
        l2  = self.head2d(f2)                        # (B,3,16,16)
        l2u = F.interpolate(l2, (H, W), mode="bilinear", align_corners=False)

        fp  = f2.mean((2, 3)).view(B, -1, 1, 1, 1).expand(-1, -1, D, H, W)
        f3  = self.back3d(p3)                        # (B,64,D,H,W)
        f   = torch.cat([fp, f3], 1)
        l3  = self.fuse(f)                           # (B,3,D,H,W)
        return l2u, l3

# ───────── TRAIN / EVAL LOOP ──────────────────────────────────────
def run_epoch(model, dl, optim=None):
    train = optim is not None
    model.train() if train else model.eval()
    tot_loss, cm, cnt = 0.0, torch.zeros(NUM_CLASSES, NUM_CLASSES, device=DEVICE), 0

    with torch.set_grad_enabled(train):
        for b in dl:
            p2, p3 = b["pv2d"].to(DEVICE), b["pv3d"].to(DEVICE)
            y2     = b.get("lbl2d")
            y3     = b.get("lbl3d")
            if y2 is not None: y2 = y2.to(DEVICE)
            if y3 is not None: y3 = y3.to(DEVICE)

            if train: optim.zero_grad()
            o2, o3 = model(p2, p3)

            if y3 is not None:
                l2  = F.cross_entropy(o2, y2, weight=CE_W_2D.to(DEVICE))
                ce3 = F.cross_entropy(o3, y3, weight=CE_W_3D.to(DEVICE))
                loss = l2 + ce3
                preds = o3.argmax(1).view(-1)
                _cm_upd(cm, preds, y3.view(-1))
            else:
                loss = torch.tensor(0.0, device=DEVICE)

            if train:
                loss.backward()
                optim.step()

            tot_loss += loss.item()
            cnt += 1

    loss_avg    = tot_loss / (cnt + 1e-7)
    acc_g, iou, f1, rec, prec, acc_c = _cm_reduce(cm)
    return loss_avg, acc_g, iou, f1, rec, prec, acc_c

# ───────── MAIN ───────────────────────────────────────────────────
def main():
    torch.cuda.empty_cache()
    gc.collect()

    ds_tr = UXBDataset(ROOT, "train")
    ds_te = UXBDataset(ROOT, "test")

    dl_tr = DataLoader(ds_tr, BATCH_SIZE,
                       sampler=BalancedSampler(ds_tr),
                       collate_fn=collate,
                       num_workers=NUM_WORKERS,
                       pin_memory=True)
    dl_te = DataLoader(ds_te, BATCH_SIZE,
                       shuffle=False,
                       collate_fn=collate,
                       num_workers=NUM_WORKERS,
                       pin_memory=True)

    model = FusedSeg().to(DEVICE)
    head_params, enc_params = [], []
    for n, p in model.named_parameters():
        if n.startswith(("back2d.", "back3d.")):
            p.requires_grad_(False)
            enc_params.append(p)
        else:
            head_params.append(p)

    optim = torch.optim.AdamW(
        [{"params": head_params, "lr": LR_MAIN},
         {"params": enc_params,  "lr": 0.0}],
        weight_decay=1e-2
    )

    def lr_head(ep):
        return (LR_WARMUP + (LR_MAIN - LR_WARMUP) * ep / WARMUP_EPOCHS) \
               if ep <= WARMUP_EPOCHS else LR_MAIN

    def lr_enc(ep):
        if ep <= WARMUP_EPOCHS:            return 0.0
        elif ep <= 2 * WARMUP_EPOCHS:
            return LR_WARMUP + (LR_MAIN - LR_WARMUP) * (ep - WARMUP_EPOCHS) / WARMUP_EPOCHS
        else:                              return LR_MAIN

    best_val, no_improve = float("inf"), 0

    for ep in range(1, EPOCHS + 1):
        hlr, elr = lr_head(ep), lr_enc(ep)
        optim.param_groups[0]["lr"], optim.param_groups[1]["lr"] = hlr, elr

        if ep == WARMUP_EPOCHS + 1:
            for p in enc_params:
                p.requires_grad_(True)
            print(f"→ Unfroze backbones at epoch {ep}")

        print(f"Epoch {ep:03d}/{EPOCHS}  head_lr={hlr:.2e}  enc_lr={elr:.2e}")
        tr = run_epoch(model, dl_tr, optim)
        ev = run_epoch(model, dl_te)

        tr_loss, tr_acc = tr[0], tr[1]
        val_loss, val_acc, iou, f1, rec, prec, acc_c = ev

        print(
            f"[{ep:03}/{EPOCHS}] "
            f"train_loss={tr_loss:.3f} train_acc={tr_acc:.3f} | "
            f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}\n"
            f"    IoU  [{_fmt(iou)}]\n"
            f"    F1   [{_fmt(f1)}]\n"
            f"    Rcl  [{_fmt(rec)}]\n"
            f"    Prc  [{_fmt(prec)}]\n"
            f"    AccC [{_fmt(acc_c)}]"
        )

        if val_loss < best_val - 1e-6:
            best_val, no_improve = val_loss, 0
            torch.save(model.state_dict(), BEST_PATH)
            print(f"✔ Saved best → {BEST_PATH}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("✘ Early stopping (no val‑loss improvement).")
                break

    # ───────── INFERENCE ───────────────────────────────────────────
    ds_pr = UXBDataset(ROOT, "predict")
    dl_pr = DataLoader(ds_pr, 1, shuffle=False, collate_fn=collate)

    out_dir = ROOT / "predict_out"
    out_dir.mkdir(exist_ok=True)

    print("\nInference:")
    model.eval()
    for i, b in enumerate(dl_pr):
        p2, p3 = b["pv2d"].to(DEVICE), b["pv3d"].to(DEVICE)
        las_path = Path(b["las_path"][0])

        with torch.no_grad():
            _, o3 = model(p2, p3)
        vol = o3.argmax(1).cpu().numpy()[0]
        np.save(out_dir / f"pred_{i:04d}.npy", vol)

        las = laspy.read(las_path)
        x, y, z = las.x, las.y, las.z
        D, H, W = GRID_SIZE
        xi = ((x - x.min()) / (x.max() - x.min()) * (W - 1)).astype(int)
        yi = ((y - y.min()) / (y.max() - y.min()) * (H - 1)).astype(int)
        zi = ((z - z.min()) / (z.max() - z.min()) * (D - 1)).astype(int)
        pred_cls = vol[zi, yi, xi]
        las.classification = np.vectorize(INV_LABEL_MAP_3D.get)(pred_cls).astype(np.uint8)
        las.write(out_dir / f"pred_{i:04d}.las")
        print(f"  saved pred_{i:04d}.npy & pred_{i:04d}.las")

if __name__ == "__main__":
    main()
