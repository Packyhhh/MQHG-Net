import os
import re
import json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from model_hypergraph_mm_plasma import HypergraphMMNet, HypergraphMMConfig
SEED = int(os.getenv("SEED", "42"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_CSV = os.getenv("DATA_CSV", "data/MASTER.csv")
ROI_THR_JSON = os.getenv("ROI_THR_JSON", "data/roi_thresholds_ab_cn.json")
ROI_THR_KEY = os.getenv("ROI_THR_KEY", "mean_plus_2sd")  # fixed thresholds in your json

SAVE_DIR = os.getenv("SAVE_DIR", "kfold_std_tauAUR_col_aligned2")

ONLY_TEST = os.environ.get("ONLY_TEST", "0") == "1"

N_FOLDS = int(os.getenv("N_FOLDS", "5"))
INNER_VAL_RATIO = float(os.getenv("INNER_VAL_RATIO", "0.15"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))

MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", "60"))
PATIENCE = int(os.getenv("PATIENCE", "12"))

LR = float(os.getenv("LR", "3e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))

# Model hparams
K_QUERIES = int(os.getenv("K_QUERY", "16"))
D_MODEL = int(os.getenv("D_MODEL", "128"))
N_LAYERS = int(os.getenv("N_HG_LAYERS", "1"))
DROPOUT = float(os.getenv("DROPOUT", "0.1"))

# loss weights
LAMBDA_AMY = float(os.getenv("LAMBDA_AMY", "1.0"))
LAMBDA_TAU_GLOBAL = float(os.getenv("LAMBDA_TAU_GLOBAL", "1.0"))
LAMBDA_TAU_ROI = float(os.getenv("LAMBDA_TAU_ROI", "1.0"))

# early-stop metric (computed on INNER-VAL)
EARLY_STOP_KEY = os.getenv("EARLY_STOP_KEY", "tau_roi_mean_AUPRC")  ##tau_roi_mean_AUPRC,tau_global_AUROC

PLASMA_COLS = ["pT217_F", "GFAP_Q", "NfL_Q", "AB42_AB40_F"]  # "GFAP_Q", "NfL_Q", "AB42_AB40_F"


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_ta_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if re.match(r"^ST\d+TA$", str(c), flags=re.IGNORECASE)]
    return sorted(cols, key=lambda x: int(re.findall(r"\d+", x)[0]))


def encode_gender(series: pd.Series) -> pd.Series:
    def f(x):
        if pd.isna(x):
            return np.nan
        t = str(x).strip().upper()
        if t.startswith("M"):
            return 1.0
        if t.startswith("F"):
            return 0.0
        return np.nan

    return series.apply(f)


def compute_mean_std(train_df: pd.DataFrame, cols: List[str]) -> Dict[str, Tuple[float, float]]:
    ms = {}
    for c in cols:
        v = pd.to_numeric(train_df[c], errors="coerce").astype(float)
        mu = float(np.nanmean(v))
        sd = float(np.nanstd(v))
        if not np.isfinite(mu):
            mu = 0.0
        if not np.isfinite(sd) or sd < 1e-8:
            sd = 1.0
        ms[c] = (mu, sd)
    return ms


def norm_array(x: np.ndarray, cols: List[str], mean_std: Dict[str, Tuple[float, float]]) -> np.ndarray:
    y = x.astype(np.float32, copy=True)
    for j, c in enumerate(cols):
        mu, sd = mean_std[c]
        y[j] = (y[j] - mu) / (sd if sd > 1e-8 else 1.0)
    return y


def load_roi_thresholds(json_path: str, key: str) -> Dict[str, float]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"ROI threshold json not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    if "thresholds" not in j or key not in j["thresholds"]:
        raise KeyError(f"Bad ROI threshold json: need thresholds['{key}']")
    raw = j["thresholds"][key]
    thr = {k: float(v) for k, v in raw.items() if v is not None and np.isfinite(v)}
    return thr


def build_roi_labels_matrix(df: pd.DataFrame, roi_cols: List[str], thr: Dict[str, float]) -> np.ndarray:
    X = df[roi_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    Y = np.full_like(X, -1.0, dtype=np.float32)
    for j, c in enumerate(roi_cols):
        t = thr.get(c, np.nan)
        if not np.isfinite(t):
            continue
        v = X[:, j]
        m = np.isfinite(v)
        Y[m, j] = (v[m] > t).astype(np.float32)
    return Y


def compute_pos_weight_binary(y: np.ndarray) -> torch.Tensor:
    y = y[np.isfinite(y)]
    y = y[(y == 0) | (y == 1)]
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return torch.tensor([neg / max(pos, 1)], device=DEVICE, dtype=torch.float32)


def compute_pos_weight_vec_multilabel(Y: np.ndarray) -> torch.Tensor:
    pos_w = []
    for r in range(Y.shape[1]):
        yr = Y[:, r]
        yr = yr[np.isfinite(yr)]
        yr = yr[(yr == 0) | (yr == 1)]
        p = int((yr == 1).sum())
        n = int((yr == 0).sum())
        pos_w.append(n / max(p, 1))
    return torch.tensor(pos_w, device=DEVICE, dtype=torch.float32)


class MTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, ta_cols: List[str], roi_cols: List[str], roi_thr: Dict[str, float],
                 aux_cols: List[str], mean_std: Dict[str, Tuple[float, float]]):
        self.df = df.reset_index(drop=True)
        self.ta_cols = ta_cols
        self.roi_cols = roi_cols
        self.roi_thr = roi_thr
        self.aux_cols = aux_cols
        self.mean_std = mean_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # MRI TA (68)
        x_mri = row[self.ta_cols].to_numpy(dtype=np.float32)
        x_mri = norm_array(x_mri, self.ta_cols, self.mean_std)
        x_mri = np.nan_to_num(x_mri, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        def get_val(name, default=np.nan):
            return row[name] if name in self.df.columns else default

        # plasma (4 cols)
        plasma_vals = []
        avail_any = False
        for c in PLASMA_COLS:
            v = get_val(c)
            if pd.notna(v) and np.isfinite(v):
                avail_any = True
                plasma_vals.append(np.float32(v))
            else:
                plasma_vals.append(np.float32(np.nan))
        avail_plasma = bool(avail_any)
        plasma = np.array(plasma_vals, dtype=np.float32)

        # apoe
        a = get_val("APOE4")
        avail_apoe = (pd.notna(a) and np.isfinite(a))
        apoe = np.array([a if avail_apoe else np.nan], dtype=np.float32)

        # demo
        age = get_val("AGE")
        edu = get_val("PTEDUCAT")
        sex = get_val("PTGENDER_BIN")
        avail_demo = (pd.notna(age) and pd.notna(edu) and pd.notna(sex)
                      and np.isfinite(age) and np.isfinite(edu) and np.isfinite(sex))
        demo = np.array([age, edu, sex], dtype=np.float32)

        # normalize aux columns
        for (arr, cols) in [
            (plasma, PLASMA_COLS),
            (apoe, ["APOE4"]),
            (demo, ["AGE", "PTEDUCAT", "PTGENDER_BIN"]),
        ]:
            for j, c in enumerate(cols):
                if c in self.mean_std:
                    mu, sd = self.mean_std[c]
                    v = arr[j]
                    if np.isfinite(v):
                        arr[j] = (v - mu) / (sd if sd > 1e-8 else 1.0)

        # Replace nan with 0 for model input
        plasma = np.nan_to_num(plasma, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        apoe = np.nan_to_num(apoe, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        demo = np.nan_to_num(demo, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Labels
        amy = get_val("amy_pos")
        y_amy = np.float32(amy) if pd.notna(amy) else np.float32(-1.0)

        tau_g = get_val("tau_pos_std")
        y_tau_g = np.float32(tau_g) if pd.notna(tau_g) else np.float32(-1.0)

        # ROI labels from SUVR + fixed thresholds
        suvr = row[self.roi_cols].to_numpy(dtype=np.float32)
        y_tau_roi = np.full((len(self.roi_cols),), -1.0, dtype=np.float32)
        for j, c in enumerate(self.roi_cols):
            v = suvr[j]
            t = self.roi_thr.get(c, np.nan)
            if np.isfinite(v) and np.isfinite(t):
                y_tau_roi[j] = 1.0 if (v > t) else 0.0

        return {
            "RID": int(row["RID"]),
            "x_mri": torch.tensor(x_mri),
            "x_plasma": torch.tensor(plasma),
            "x_apoe": torch.tensor(apoe),
            "x_demo": torch.tensor(demo),
            "avail_plasma": torch.tensor(bool(avail_plasma)),
            "avail_apoe": torch.tensor(bool(avail_apoe)),
            "avail_demo": torch.tensor(bool(avail_demo)),
            "y_amy": torch.tensor(y_amy),
            "y_tau_g": torch.tensor(y_tau_g),
            "y_tau_roi": torch.tensor(y_tau_roi),
        }


def masked_bce_logits(logits: torch.Tensor, targets: torch.Tensor, missing_value: float = -1.0):
    mask = targets != missing_value
    if mask.sum() == 0:
        return logits.sum() * 0.0
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    return loss_fn(logits[mask], targets[mask])


def multilabel_masked_bce(logits: torch.Tensor, targets: torch.Tensor, missing_value: float = -1.0, pos_weight_vec=None):
    mask = targets != missing_value
    if mask.sum() == 0:
        return logits.sum() * 0.0
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_vec, reduction="none")
    loss = loss_fn(logits, targets.clamp(min=0.0))
    return loss[mask].mean()


@torch.no_grad()
def eval_fold(model, loader):
    model.eval()
    y_amy, p_amy = [], []
    y_tau, p_tau = [], []

    roi_targets = []
    roi_probs = []

    for batch in loader:
        x_mri = batch["x_mri"].to(DEVICE)
        x_plasma = batch["x_plasma"].to(DEVICE)
        x_apoe = batch["x_apoe"].to(DEVICE)
        x_demo = batch["x_demo"].to(DEVICE)
        av_p = batch["avail_plasma"].to(DEVICE)
        av_a = batch["avail_apoe"].to(DEVICE)
        av_d = batch["avail_demo"].to(DEVICE)

        out = model(
            x_mri, x_plasma, x_apoe, x_demo,
            avail_plasma=av_p, avail_apoe=av_a, avail_demo=av_d,
            return_attn=False
        )

        # amy
        ya = batch["y_amy"].cpu().numpy()
        pa = torch.sigmoid(out["amy_logit"]).detach().cpu().numpy()
        m = ya != -1
        if m.sum() > 0:
            y_amy.append(ya[m].astype(int))
            p_amy.append(pa[m])

        # tau global
        yt = batch["y_tau_g"].cpu().numpy()
        pt = torch.sigmoid(out["tau_global_logit"]).detach().cpu().numpy()
        m = yt != -1
        if m.sum() > 0:
            y_tau.append(yt[m].astype(int))
            p_tau.append(pt[m])

        # roi
        y = batch["y_tau_roi"].cpu().numpy().astype(np.float32)
        p = torch.sigmoid(out["tau_roi_logits"]).cpu().numpy().astype(np.float32)
        roi_targets.append(y)
        roi_probs.append(p)

    metrics = {}

    if len(y_amy) > 0:
        ya = np.concatenate(y_amy)
        pa = np.concatenate(p_amy)
        if len(np.unique(ya)) > 1:
            metrics["amy_AUROC"] = float(roc_auc_score(ya, pa))
            metrics["amy_AUPRC"] = float(average_precision_score(ya, pa))
            metrics["amy_ACC"] = float(accuracy_score(ya, (pa >= 0.5).astype(int)))
        metrics["amy_n"] = int(len(ya))

    if len(y_tau) > 0:
        yt = np.concatenate(y_tau)
        pt = np.concatenate(p_tau)
        if len(np.unique(yt)) > 1:
            metrics["tau_global_AUROC"] = float(roc_auc_score(yt, pt))
            metrics["tau_global_AUPRC"] = float(average_precision_score(yt, pt))
            metrics["tau_global_ACC"] = float(accuracy_score(yt, (pt >= 0.5).astype(int)))
        metrics["tau_global_n"] = int(len(yt))

    Y = np.concatenate(roi_targets, axis=0) if len(roi_targets) else np.zeros((0, 0), np.float32)
    P = np.concatenate(roi_probs, axis=0) if len(roi_probs) else np.zeros((0, 0), np.float32)

    aurocs, auprcs = [], []
    valid_rois = 0
    for r in range(Y.shape[1]):
        yr = Y[:, r]
        pr = P[:, r]
        m = yr != -1
        if m.sum() == 0:
            continue
        yr = yr[m].astype(int)
        pr = pr[m]
        if (yr == 1).sum() < 10 or (yr == 0).sum() < 10:
            continue
        if len(np.unique(yr)) > 1:
            aurocs.append(float(roc_auc_score(yr, pr)))
            auprcs.append(float(average_precision_score(yr, pr)))
            valid_rois += 1

    metrics["tau_roi_valid_rois"] = int(valid_rois)
    metrics["tau_roi_mean_AUROC"] = float(np.mean(aurocs)) if len(aurocs) else np.nan
    metrics["tau_roi_mean_AUPRC"] = float(np.mean(auprcs)) if len(auprcs) else np.nan

    return metrics, Y, P

def save_global_preds_npz(model, dl, out_npz_path: str):
    """
    Save per-sample predictions for the *entire* dataloader length (same as test_split.csv rows).
    Do NOT drop rows with missing labels. Missing labels stay as -1.
    This prevents length mismatch when aligning with CSV.
    """
    import numpy as np
    import torch

    model.eval()

    rids = []
    y_amy_all, p_amy_all = [], []
    y_tau_all, p_tau_all = [], []

    with torch.no_grad():
        for batch in dl:
            # --------- RID (must align with CSV order) ----------
            if "RID" in batch:
                rid_b = batch["RID"].detach().cpu().numpy()
            elif "rid" in batch:
                rid_b = batch["rid"].detach().cpu().numpy()
            else:
                # fallback: no RID in batch, still keep alignment by position
                rid_b = np.full((batch["x_mri"].shape[0],), -1, dtype=np.int64)
            rids.append(rid_b.astype(np.int64))

            # --------- forward ----------
            x_mri = batch["x_mri"].to(DEVICE)
            x_plasma = batch["x_plasma"].to(DEVICE)
            x_apoe = batch["x_apoe"].to(DEVICE)
            x_demo = batch["x_demo"].to(DEVICE)
            av_p = batch["avail_plasma"].to(DEVICE)
            av_a = batch["avail_apoe"].to(DEVICE)
            av_d = batch["avail_demo"].to(DEVICE)

            out = model(
                x_mri, x_plasma, x_apoe, x_demo,
                avail_plasma=av_p, avail_apoe=av_a, avail_demo=av_d,
                return_attn=False
            )

            # --------- labels (keep -1 as missing) ----------
            # amy
            if "y_amy" in batch:
                y_amy_b = batch["y_amy"].detach().cpu().numpy().astype(np.int32)
            else:
                y_amy_b = np.full((x_mri.shape[0],), -1, dtype=np.int32)

            # tau-global label key in your code is y_tau_g
            if "y_tau_g" in batch:
                y_tau_b = batch["y_tau_g"].detach().cpu().numpy().astype(np.int32)
            elif "y_tau" in batch:
                y_tau_b = batch["y_tau"].detach().cpu().numpy().astype(np.int32)
            else:
                y_tau_b = np.full((x_mri.shape[0],), -1, dtype=np.int32)

            y_amy_all.append(y_amy_b)
            y_tau_all.append(y_tau_b)

            # --------- probs (compute for all rows, no filtering) ----------
            # amy prob
            if "amy_logit" in out:
                p_amy_b = torch.sigmoid(out["amy_logit"]).detach().cpu().numpy().astype(np.float32)
            else:
                p_amy_b = np.full((x_mri.shape[0],), np.nan, dtype=np.float32)

            # tau prob (global)
            if "tau_global_logit" in out:
                p_tau_b = torch.sigmoid(out["tau_global_logit"]).detach().cpu().numpy().astype(np.float32)
            else:
                p_tau_b = np.full((x_mri.shape[0],), np.nan, dtype=np.float32)

            p_amy_all.append(p_amy_b)
            p_tau_all.append(p_tau_b)

    rid = np.concatenate(rids, axis=0)
    y_amy = np.concatenate(y_amy_all, axis=0)
    p_amy = np.concatenate(p_amy_all, axis=0)
    y_tau = np.concatenate(y_tau_all, axis=0)
    p_tau = np.concatenate(p_tau_all, axis=0)

    np.savez_compressed(
        out_npz_path,
        rid=rid,
        y_amy=y_amy, p_amy=p_amy,
        y_tau=y_tau, p_tau=p_tau,
    )
    print(f"[OK] Saved global preds (aligned): {out_npz_path}  N={len(rid)}")

def save_roi_preds_npz(model, loader, out_npz_path: str, roi_cols=None):
    model.eval()
    Ys, Ps = [], []

    with torch.no_grad():
        for batch in loader:
            x_mri = batch["x_mri"].to(DEVICE)
            x_plasma = batch["x_plasma"].to(DEVICE)
            x_apoe = batch["x_apoe"].to(DEVICE)
            x_demo = batch["x_demo"].to(DEVICE)
            av_p = batch["avail_plasma"].to(DEVICE)
            av_a = batch["avail_apoe"].to(DEVICE)
            av_d = batch["avail_demo"].to(DEVICE)

            out = model(
                x_mri, x_plasma, x_apoe, x_demo,
                avail_plasma=av_p, avail_apoe=av_a, avail_demo=av_d,
                return_attn=False
            )

            y = batch["y_tau_roi"].cpu().numpy().astype(np.float32)
            p = torch.sigmoid(out["tau_roi_logits"]).cpu().numpy().astype(np.float32)
            Ys.append(y)
            Ps.append(p)

    Y = np.concatenate(Ys, axis=0) if len(Ys) else np.zeros((0, 0), np.float32)
    P = np.concatenate(Ps, axis=0) if len(Ps) else np.zeros((0, 0), np.float32)

    if roi_cols is not None:
        assert len(roi_cols) == Y.shape[1], f"len(roi_cols)={len(roi_cols)} but Y.shape[1]={Y.shape[1]}"

        np.savez_compressed(
            out_npz_path,
            y_roi=Y,
            p_roi=P,
            roi_cols=np.array(roi_cols, dtype=object),
        )
    else:
        np.savez_compressed(out_npz_path, y_roi=Y, p_roi=P)

    print("[OK] Saved ROI preds:", out_npz_path)

def mean_std_summary(metrics_list: List[Dict[str, float]]):
    keys = sorted({k for m in metrics_list for k in m.keys()})
    out = {}
    for k in keys:
        vals = [float(m.get(k, np.nan)) for m in metrics_list]
        arr = np.array(vals, dtype=float)
        out[k] = [float(np.nanmean(arr)), float(np.nanstd(arr, ddof=0))]
    return out


def main():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"[INFO] SAVE_DIR={SAVE_DIR}")
    print(f"[INFO] K_QUERIES={K_QUERIES} (cfg.n_query)")
    print(f"[INFO] ROI_THR_JSON={ROI_THR_JSON} key={ROI_THR_KEY}")
    print(f"[INFO] Plasma cols={PLASMA_COLS}")

    df = pd.read_csv(DATA_CSV, low_memory=False)

    # tau global label: tau_pos_std
    df["tau_pos_std"] = pd.to_numeric(df["tau_pos_std"], errors="coerce")
    before = len(df)
    df = df[df["tau_pos_std"].notna()].copy()

    after = len(df)
    if after < before:
        print(f"[WARN] Dropped {before - after} rows with missing tau_pos_std.")
    df["tau_pos_std"] = df["tau_pos_std"].astype(int)

    # gender bin
    if "PTGENDER" in df.columns and "PTGENDER_BIN" not in df.columns:
        df["PTGENDER_BIN"] = encode_gender(df["PTGENDER"])
    if "PTGENDER_BIN" not in df.columns:
        df["PTGENDER_BIN"] = np.nan

    # ta_cols = pick_ta_cols(df)
    # ====== USE ALIGNED MRI ROI ORDER ======
    ta_align_path = os.path.join("data", "ta_cols_aligned.txt")

    if os.path.exists(ta_align_path):
        ta_cols = [ln.strip() for ln in open(ta_align_path, "r", encoding="utf-8") if ln.strip()]
        assert len(ta_cols) == 68, f"Aligned ta_cols must be 68, got {len(ta_cols)}"
        print("[INFO] Using aligned ta_cols from:", ta_align_path)
    else:
        raise RuntimeError("Aligned ROI file not found. You must run alignment first.")

    assert len(ta_cols) == 68, f"Expected 68 MRI TA cols, got {len(ta_cols)}"
    assert "amy_pos" in df.columns, "Missing amy_pos"
    assert "RID" in df.columns, "Missing RID"

    # fixed ROI thresholds
    roi_thr = load_roi_thresholds(ROI_THR_JSON, ROI_THR_KEY)
    roi_cols = sorted(list(roi_thr.keys()))
    missing_roi_cols = [c for c in roi_cols if c not in df.columns]
    if missing_roi_cols:
        raise RuntimeError(f"ROI SUVR columns missing in master: {missing_roi_cols[:10]} ... total={len(missing_roi_cols)}")

    # plasma columns must exist
    missing_plasma = [c for c in PLASMA_COLS if c not in df.columns]
    if missing_plasma:
        raise RuntimeError(f"Plasma columns missing in master: {missing_plasma}")

    aux_cols = PLASMA_COLS + ["APOE4", "AGE", "PTEDUCAT", "PTGENDER_BIN"]

    groups_all = df["RID"].astype(int).to_numpy()
    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = list(gkf.split(df, groups=groups_all))

    all_test_metrics = []

    for fold_i, (outer_tr_idx, outer_te_idx) in enumerate(folds, start=1):
        fold_dir = os.path.join(SAVE_DIR, f"fold{fold_i}")
        os.makedirs(fold_dir, exist_ok=True)

        # ========== inside: for fold_i, (outer_tr_idx, outer_te_idx) in enumerate(folds, start=1): ==========

        fold_dir = os.path.join(SAVE_DIR, f"fold{fold_i}")
        os.makedirs(fold_dir, exist_ok=True)

        train_csv = os.path.join(fold_dir, "train_split.csv")
        val_csv = os.path.join(fold_dir, "val_split.csv")
        test_csv = os.path.join(fold_dir, "test_split.csv")

        # ---- 1) split：训练时生成；ONLY_TEST 时复用已有 split（不再重划分）----
        if ONLY_TEST:
            if not (os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv)):
                raise RuntimeError(
                    f"[ONLY_TEST] Missing split csv in {fold_dir}.\n"
                    f"Need existing train_split.csv/val_split.csv/test_split.csv from a previous training run."
                )

            tr = pd.read_csv(train_csv, low_memory=False)
            va = pd.read_csv(val_csv, low_memory=False)
            outer_te = pd.read_csv(test_csv, low_memory=False)

        else:
            outer_tr = df.iloc[outer_tr_idx].copy()
            outer_te = df.iloc[outer_te_idx].copy()

            gss = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_RATIO, random_state=SEED + fold_i)
            tr_idx, va_idx = next(gss.split(outer_tr, groups=outer_tr["RID"].astype(int)))
            tr = outer_tr.iloc[tr_idx].copy()
            va = outer_tr.iloc[va_idx].copy()

            tr.to_csv(train_csv, index=False)
            va.to_csv(val_csv, index=False)
            outer_te.to_csv(test_csv, index=False)

        # ---- 2) mean/std：训练时算；ONLY_TEST 时从 best.pt 读（不再重算）----
        if ONLY_TEST:
            best_path = os.path.join(fold_dir, "best.pt")
            if not os.path.exists(best_path):
                raise RuntimeError(f"[ONLY_TEST] Missing {best_path}")

            ckpt = torch.load(best_path, map_location="cpu")
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

            if not (isinstance(ckpt, dict) and "mean_std" in ckpt):
                raise RuntimeError("[ONLY_TEST] best.pt has no 'mean_std'. You must have saved it during training.")
            mean_std_use = ckpt["mean_std"]

        else:
            mean_std_use = compute_mean_std(tr, ta_cols + aux_cols)


        if ONLY_TEST:
            ds_te = MTDataset(outer_te, ta_cols, roi_cols, roi_thr, aux_cols, mean_std_use)
            dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, drop_last=False)


            cfg_dict = ckpt["cfg"]
            cfg = HypergraphMMConfig(**cfg_dict)
            model = HypergraphMMNet(cfg).to(DEVICE)
            model.load_state_dict(state_dict)

            test_metrics, _, _ = eval_fold(model, dl_te)

            with open(os.path.join(fold_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, ensure_ascii=False, indent=2)


            save_roi_preds_npz(model, dl_te, os.path.join(fold_dir, "roi_preds_test.npz"), roi_cols=roi_cols)
            save_global_preds_npz(model, dl_te, os.path.join(fold_dir, "global_preds_test.npz"))

            all_test_metrics.append(test_metrics)
            continue

        else:
            ds_tr = MTDataset(tr, ta_cols, roi_cols, roi_thr, aux_cols, mean_std_use)
            ds_va = MTDataset(va, ta_cols, roi_cols, roi_thr, aux_cols, mean_std_use)
            ds_te = MTDataset(outer_te, ta_cols, roi_cols, roi_thr, aux_cols, mean_std_use)

            dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, drop_last=False)
            dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, drop_last=False)
            dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, drop_last=False)


        pos_w_global = compute_pos_weight_binary(tr["tau_pos_std"].to_numpy(dtype=float))
        Y_tr_roi = build_roi_labels_matrix(tr, roi_cols, roi_thr)
        pos_w_roi = compute_pos_weight_vec_multilabel(Y_tr_roi)



        # ---- build model (TRAIN branch) ----
        cfg = HypergraphMMConfig(
            n_roi=68,
            n_query=K_QUERIES,
            d_model=D_MODEL,
            n_hg_layers=N_LAYERS,
            dropout=DROPOUT,
            d_plasma_in=len(PLASMA_COLS),
        )
        model = HypergraphMMNet(cfg).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best = None
        bad = 0

        for ep in range(1, MAX_EPOCHS + 1):
            model.train()
            losses = []

            for batch in dl_tr:
                x_mri = batch["x_mri"].to(DEVICE)
                x_plasma = batch["x_plasma"].to(DEVICE)
                x_apoe = batch["x_apoe"].to(DEVICE)
                x_demo = batch["x_demo"].to(DEVICE)
                av_p = batch["avail_plasma"].to(DEVICE)
                av_a = batch["avail_apoe"].to(DEVICE)
                av_d = batch["avail_demo"].to(DEVICE)

                y_amy = batch["y_amy"].to(DEVICE)
                y_tau_g = batch["y_tau_g"].to(DEVICE)
                y_tau_roi = batch["y_tau_roi"].to(DEVICE)

                out = model(
                    x_mri, x_plasma, x_apoe, x_demo,
                    avail_plasma=av_p, avail_apoe=av_a, avail_demo=av_d,
                    return_attn=False
                )

                loss_amy = masked_bce_logits(out["amy_logit"], y_amy, missing_value=-1.0)

                m = (y_tau_g != -1.0)
                if m.sum() > 0:
                    loss_tau_g = nn.BCEWithLogitsLoss(pos_weight=pos_w_global)(out["tau_global_logit"][m], y_tau_g[m])
                else:
                    loss_tau_g = out["tau_global_logit"].sum() * 0.0

                loss_tau_roi = multilabel_masked_bce(out["tau_roi_logits"], y_tau_roi, missing_value=-1.0,
                                                    pos_weight_vec=pos_w_roi)

                loss = LAMBDA_AMY * loss_amy + LAMBDA_TAU_GLOBAL * loss_tau_g + LAMBDA_TAU_ROI * loss_tau_roi

                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))

            val_metrics, _, _ = eval_fold(model, dl_va)
            key = val_metrics.get(EARLY_STOP_KEY, None)
            key_val = -1.0 if key is None or not np.isfinite(key) else float(key)

            if best is None or key_val > best["key"]:
                best = {"key": key_val, "epoch": ep, "metrics": val_metrics}
                bad = 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "cfg": cfg.__dict__,
                        "mean_std": mean_std_use,
                        "ta_cols": ta_cols,
                        "roi_cols": roi_cols,
                        "roi_thr": roi_thr,
                        "plasma_cols": PLASMA_COLS,
                        "aux_cols": aux_cols,
                        "best": best,
                    },
                    os.path.join(fold_dir, "best.pt"),
                )
            else:
                bad += 1

            print(f"[Fold {fold_i}] ep={ep:03d} loss={np.mean(losses):.4f} {EARLY_STOP_KEY}={key_val:.4f} bad={bad}/{PATIENCE}")

            if bad >= PATIENCE:
                break

        ckpt = torch.load(os.path.join(fold_dir, "best.pt"), map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        model.to(DEVICE)

        test_metrics, _, _ = eval_fold(model, dl_te)

        with open(os.path.join(fold_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fold": fold_i,
                    "best": ckpt["best"],
                    "test_metrics": test_metrics,
                    "EARLY_STOP_KEY": EARLY_STOP_KEY,
                    "ROI_THR_JSON": ROI_THR_JSON,
                    "ROI_THR_KEY": ROI_THR_KEY,
                    "plasma_cols": PLASMA_COLS,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        with open(os.path.join(fold_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)

        save_roi_preds_npz(model, dl_te, os.path.join(fold_dir, "roi_preds_test.npz"), roi_cols=roi_cols)
        save_global_preds_npz(model, dl_te, os.path.join(fold_dir, "global_preds_test.npz"))

        all_test_metrics.append(test_metrics)
        print(f"[Fold {fold_i}] TEST tau_roi_mean_AUPRC={test_metrics.get('tau_roi_mean_AUPRC')} tau_global_AUROC={test_metrics.get('tau_global_AUROC')}")

    summary = mean_std_summary(all_test_metrics)
    with open(os.path.join(SAVE_DIR, "summary_test.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[DONE] Wrote:", os.path.join(SAVE_DIR, "summary_test.json"))


if __name__ == "__main__":
    main()