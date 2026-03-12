# cluster_pd_routeA.py
# -*- coding: utf-8 -*-

import os
import math
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from scipy import stats
from scipy.signal import welch, hilbert, find_peaks

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import f_classif
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering

import matplotlib.pyplot as plt

# Optional deps
try:
    import pywt  # pip install PyWavelets
    HAS_PYWT = True
except Exception:
    HAS_PYWT = False

try:
    import umap  # pip install umap-learn
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    import hdbscan  # pip install hdbscan
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

try:
    from joblib import Parallel, delayed  # pip install joblib
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False


# ----------------------------
# Config
# ----------------------------
@dataclass
class FeatureConfig:
    fs: float = 62.5e6 # sampling rate (Hz)
    # Welch PSD
    welch_nperseg: int = 1024
    welch_noverlap: Optional[int] = None
    # Peak/echo detection
    peak_prominence_q: float = 0.90  # use quantile of envelope to set prominence/height
    min_peak_distance_ms: float = 0.05  # min distance between peaks (ms)
    max_echo_peaks: int = 8
    # Wavelet
    wavelet_name: str = "db4"
    wavelet_levels: int = 6
    # Normalization
    robust_center: bool = True
    robust_scale: bool = True


@dataclass
class ClusterConfig:
    reducer: str = "pca"  # "pca" or "umap"
    pca_variance: float = 0.95
    umap_n_neighbors: int = 25
    umap_min_dist: float = 0.05
    umap_n_components: int = 2

    # Density clustering
    density_method: str = "hdbscan"  # "hdbscan" / "optics" / "dbscan"
    hdb_min_cluster_size: int = 20
    hdb_min_samples: Optional[int] = None

    optics_min_samples: int = 20
    optics_xi: float = 0.05
    optics_min_cluster_size: int = 0.03  # can be fraction or int

    dbscan_eps: float = 0.6
    dbscan_min_samples: int = 20

    # Hierarchical clustering (optional)
    do_hierarchical: bool = False
    hier_n_clusters: int = 10
    hier_linkage: str = "ward"  # "ward"/"average"/"complete"/"single"


# ----------------------------
# Data loading helpers
# ----------------------------
def load_signals_from_parquet(
    path: str,
    signal_col: str = "signal",
    id_col: Optional[str] = None,
    int_to_volt: bool = False,
    scale: float = 1.0,
) -> Tuple[List[np.ndarray], List[Any]]:
    """
    Expect parquet with a column containing array-like signals.
    - int_to_volt: if True, apply conversion: x * scale
      (You should set scale accordingly, e.g., LSB to volt)
    """
    df = pd.read_parquet(path)
    if signal_col not in df.columns:
        raise ValueError(f"signal_col '{signal_col}' not found in parquet columns: {df.columns.tolist()}")

    ids = df[id_col].tolist() if (id_col and id_col in df.columns) else list(range(len(df)))
    sigs_raw = df[signal_col].tolist()

    signals = []
    for s in sigs_raw:
        x = np.asarray(s, dtype=np.float64)
        if int_to_volt:
            x = x * float(scale)
        signals.append(x)
    return signals, ids


# ----------------------------
# Feature engineering
# ----------------------------
def _robust_norm(x: np.ndarray) -> np.ndarray:
    """Robust normalize: subtract median, divide by MAD (avoid zero)."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-12:
        mad = np.std(x) if np.std(x) > 1e-12 else 1.0
    return (x - med) / mad


def _time_domain_features(x: np.ndarray) -> Dict[str, float]:
    """Basic waveform statistics (both amplitude-sensitive and shape-sensitive)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 4:
        return {k: np.nan for k in [
            "len", "mean", "std", "rms", "ptp", "abs_mean", "abs_std",
            "skew", "kurtosis", "zcr", "crest", "impulse", "shape", "clearance"
        ]}

    mean = float(np.mean(x))
    std = float(np.std(x))
    rms = float(np.sqrt(np.mean(x**2)))
    ptp = float(np.ptp(x))
    absx = np.abs(x)
    abs_mean = float(np.mean(absx))
    abs_std = float(np.std(absx))

    skew = float(stats.skew(x, bias=False))
    kurt = float(stats.kurtosis(x, fisher=True, bias=False))

    # Zero-crossing rate (per sample)
    zc = np.mean(x[:-1] * x[1:] < 0.0)

    peak = float(np.max(absx))
    # Crest factor, impulse factor, shape factor, clearance factor (condition monitoring style)
    crest = peak / (rms + 1e-12)
    impulse = peak / (abs_mean + 1e-12)
    shape = rms / (abs_mean + 1e-12)
    clearance = peak / ((np.mean(np.sqrt(absx)) + 1e-12) ** 2)

    return {
        "len": float(n),
        "mean": mean,
        "std": std,
        "rms": rms,
        "ptp": ptp,
        "abs_mean": abs_mean,
        "abs_std": abs_std,
        "skew": skew,
        "kurtosis": kurt,
        "zcr": float(zc),
        "crest": float(crest),
        "impulse": float(impulse),
        "shape": float(shape),
        "clearance": float(clearance),
    }


def _echo_peak_features(x: np.ndarray, cfg: FeatureConfig) -> Dict[str, float]:
    """
    Echo-sensitive features:
    - Envelope peaks stats: number of peaks, top2 time gap, amp ratio, etc.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 8:
        return {k: np.nan for k in [
            "env_peak_count", "env_top1_amp", "env_top2_amp",
            "env_top1_top2_ratio", "env_top1_top2_dt_ms",
            "env_peak_amp_mean", "env_peak_amp_std"
        ]}

    env = np.abs(hilbert(x))
    # threshold based on quantile
    q = np.quantile(env, cfg.peak_prominence_q)
    min_dist = max(1, int((cfg.min_peak_distance_ms / 1000.0) * cfg.fs))

    peaks, props = find_peaks(env, height=q, distance=min_dist)
    if len(peaks) == 0:
        return {
            "env_peak_count": 0.0,
            "env_top1_amp": 0.0,
            "env_top2_amp": 0.0,
            "env_top1_top2_ratio": np.nan,
            "env_top1_top2_dt_ms": np.nan,
            "env_peak_amp_mean": 0.0,
            "env_peak_amp_std": 0.0,
        }

    amps = props["peak_heights"]
    # sort peaks by amplitude
    order = np.argsort(amps)[::-1]
    amps_sorted = amps[order]
    peaks_sorted = peaks[order]

    top1_amp = float(amps_sorted[0])
    top2_amp = float(amps_sorted[1]) if len(amps_sorted) > 1 else 0.0
    ratio = (top1_amp / (top2_amp + 1e-12)) if top2_amp > 0 else np.nan

    # time gap between top1 and top2 (ms)
    if len(peaks_sorted) > 1:
        dt = abs(int(peaks_sorted[0]) - int(peaks_sorted[1])) / cfg.fs * 1000.0
    else:
        dt = np.nan

    # peak amplitude summary (cap count for robustness)
    amps_cap = amps_sorted[:cfg.max_echo_peaks]
    return {
        "env_peak_count": float(len(peaks)),
        "env_top1_amp": top1_amp,
        "env_top2_amp": top2_amp,
        "env_top1_top2_ratio": float(ratio) if np.isfinite(ratio) else np.nan,
        "env_top1_top2_dt_ms": float(dt) if np.isfinite(dt) else np.nan,
        "env_peak_amp_mean": float(np.mean(amps_cap)),
        "env_peak_amp_std": float(np.std(amps_cap)),
    }


def _spectral_features(x: np.ndarray, cfg: FeatureConfig) -> Dict[str, float]:
    """Welch PSD based spectral features + bandpower ratios."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 16:
        return {k: np.nan for k in [
            "spec_centroid", "spec_bw", "spec_entropy", "spec_flatness",
            "bp_0_0p1", "bp_0p1_0p2", "bp_0p2_0p4", "bp_0p4_0p6", "bp_0p6_0p8", "bp_0p8_1p0"
        ]}

    nperseg = min(cfg.welch_nperseg, n)
    noverlap = cfg.welch_noverlap
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = min(noverlap, nperseg - 1)

    f, Pxx = welch(x, fs=cfg.fs, nperseg=nperseg, noverlap=noverlap, detrend="constant", scaling="density")
    Pxx = np.maximum(Pxx, 1e-20)

    # Normalize PSD to a probability mass for entropy/centroid
    p = Pxx / np.sum(Pxx)
    centroid = float(np.sum(f * p))
    bw = float(np.sqrt(np.sum(((f - centroid) ** 2) * p)))

    entropy = float(-np.sum(p * np.log(p + 1e-20)))
    flatness = float(np.exp(np.mean(np.log(Pxx))) / (np.mean(Pxx) + 1e-20))

    # Bandpower ratios by fraction of Nyquist
    nyq = cfg.fs / 2.0
    bands = [
        (0.0, 0.1), (0.1, 0.2), (0.2, 0.4),
        (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)
    ]
    total_power = float(np.trapz(Pxx, f))
    out = {
        "spec_centroid": centroid,
        "spec_bw": bw,
        "spec_entropy": entropy,
        "spec_flatness": flatness,
    }
    for (a, b) in bands:
        fmin, fmax = a * nyq, b * nyq
        mask = (f >= fmin) & (f <= fmax)
        bp = float(np.trapz(Pxx[mask], f[mask])) if np.any(mask) else 0.0
        out[f"bp_{a:g}_{b:g}"] = bp / (total_power + 1e-20)
    return out


def _wavelet_features(x: np.ndarray, cfg: FeatureConfig) -> Dict[str, float]:
    """
    Multi-level DWT energy features (fixed-length by padding missing levels with 0).
    Use normalized energies + per-level entropy-like measure.
    """
    if not HAS_PYWT:
        # If pywt not available, return NaNs with correct keys
        out = {}
        for i in range(1, cfg.wavelet_levels + 1):
            out[f"w_energy_L{i}"] = np.nan
        out["w_energy_A"] = np.nan
        out["w_energy_entropy"] = np.nan
        return out

    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 16:
        out = {}
        for i in range(1, cfg.wavelet_levels + 1):
            out[f"w_energy_L{i}"] = np.nan
        out["w_energy_A"] = np.nan
        out["w_energy_entropy"] = np.nan
        return out

    wavelet = pywt.Wavelet(cfg.wavelet_name)
    max_level = pywt.dwt_max_level(data_len=n, filter_len=wavelet.dec_len)
    level = min(cfg.wavelet_levels, max_level) if max_level >= 1 else 1

    coeffs = pywt.wavedec(x, wavelet=cfg.wavelet_name, level=level)
    # coeffs: [cA_level, cD_level, cD_(level-1), ... cD1]
    cA = coeffs[0]
    details = coeffs[1:]

    energies = []
    for cD in details:
        energies.append(float(np.sum(np.square(cD))))
    eA = float(np.sum(np.square(cA)))

    # pad to cfg.wavelet_levels
    # details energies currently length == level, from L=level..1
    # We'll map to L1..Lcfg (L1 is highest-frequency detail)
    # details order: cD_level (lowest-freq detail) ... cD1 (highest-freq detail)
    # So reverse to get L1..Llevel
    energies_L = list(reversed(energies))
    if len(energies_L) < cfg.wavelet_levels:
        energies_L = energies_L + [0.0] * (cfg.wavelet_levels - len(energies_L))
    else:
        energies_L = energies_L[:cfg.wavelet_levels]

    total = eA + sum(energies_L) + 1e-20
    energies_norm = [e / total for e in energies_L]
    eA_norm = eA / total

    # entropy over wavelet subbands (details + approx)
    p = np.array(energies_norm + [eA_norm], dtype=np.float64)
    p = p / (p.sum() + 1e-20)
    wentropy = float(-np.sum(p * np.log(p + 1e-20)))

    out = {f"w_energy_L{i+1}": float(energies_norm[i]) for i in range(cfg.wavelet_levels)}
    out["w_energy_A"] = float(eA_norm)
    out["w_energy_entropy"] = wentropy
    return out


def extract_features_one(x: np.ndarray, fcfg: FeatureConfig) -> Dict[str, float]:
    """
    Extract a fixed-length feature dict for one variable-length segment.
    We compute both raw amplitude features and robust-normalized features to capture
    shape vs magnitude separately.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {"len": 0.0}

    # raw
    feats = {}
    feats.update({f"raw_{k}": v for k, v in _time_domain_features(x).items()})
    feats.update({f"raw_{k}": v for k, v in _echo_peak_features(x, fcfg).items()})
    feats.update({f"raw_{k}": v for k, v in _spectral_features(x, fcfg).items()})
    feats.update({f"raw_{k}": v for k, v in _wavelet_features(x, fcfg).items()})

    # robust normalized (shape-oriented)
    xn = _robust_norm(x)
    feats.update({f"norm_{k}": v for k, v in _time_domain_features(xn).items()})
    feats.update({f"norm_{k}": v for k, v in _echo_peak_features(xn, fcfg).items()})
    feats.update({f"norm_{k}": v for k, v in _spectral_features(xn, fcfg).items()})
    feats.update({f"norm_{k}": v for k, v in _wavelet_features(xn, fcfg).items()})

    return feats


def extract_features(
    signals: List[np.ndarray],
    fcfg: FeatureConfig,
    n_jobs: int = 1
) -> pd.DataFrame:
    if HAS_JOBLIB and n_jobs != 1:
        rows = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(extract_features_one)(x, fcfg) for x in signals
        )
    else:
        rows = [extract_features_one(x, fcfg) for x in signals]
    df = pd.DataFrame(rows)
    return df


# ----------------------------
# Dimensionality reduction
# ----------------------------
def reduce_features(X: np.ndarray, ccfg: ClusterConfig, random_state: int = 42) -> Tuple[np.ndarray, Dict[str, Any]]:
    info = {}
    if ccfg.reducer.lower() == "umap":
        if not HAS_UMAP:
            print("[WARN] umap-learn not installed, fallback to PCA(2D).")
            ccfg.reducer = "pca"

    if ccfg.reducer.lower() == "umap":
        reducer = umap.UMAP(
            n_neighbors=ccfg.umap_n_neighbors,
            min_dist=ccfg.umap_min_dist,
            n_components=ccfg.umap_n_components,
            random_state=random_state,
        )
        Z = reducer.fit_transform(X)
        info["reducer"] = "umap"
        info["umap_params"] = {
            "n_neighbors": ccfg.umap_n_neighbors,
            "min_dist": ccfg.umap_min_dist,
            "n_components": ccfg.umap_n_components,
        }
        return Z, info

    # PCA path
    # 1) PCA for variance retention (for clustering)
    pca = PCA(n_components=ccfg.pca_variance, random_state=random_state)
    Xp = pca.fit_transform(X)
    info["reducer"] = "pca"
    info["pca_n_components_95var"] = int(Xp.shape[1])
    info["pca_explained_variance_sum"] = float(np.sum(pca.explained_variance_ratio_))

    # 2) For visualization we can also generate 2D PCA
    pca2 = PCA(n_components=2, random_state=random_state)
    Z2 = pca2.fit_transform(X)
    info["pca2_explained_variance_sum"] = float(np.sum(pca2.explained_variance_ratio_))
    # Return both: clustering space (Xp) and viz space (Z2)
    # Here we return Xp by default; visualization uses separate call.
    return Xp, info


def pca_2d_for_viz(X: np.ndarray, random_state: int = 42) -> np.ndarray:
    pca2 = PCA(n_components=2, random_state=random_state)
    return pca2.fit_transform(X)


# ----------------------------
# Clustering
# ----------------------------
def density_cluster(Z: np.ndarray, ccfg: ClusterConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    method = ccfg.density_method.lower()
    info = {"density_method": method}

    if method == "hdbscan":
        if HAS_HDBSCAN:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=ccfg.hdb_min_cluster_size,
                min_samples=ccfg.hdb_min_samples,
            )
            labels = clusterer.fit_predict(Z)
            info["hdbscan_params"] = {
                "min_cluster_size": ccfg.hdb_min_cluster_size,
                "min_samples": ccfg.hdb_min_samples,
            }
            # optional probability
            if hasattr(clusterer, "probabilities_"):
                info["hdbscan_prob_mean"] = float(np.mean(clusterer.probabilities_))
            return labels, info
        else:
            print("[WARN] hdbscan not installed, fallback to OPTICS.")
            method = "optics"

    if method == "optics":
        clusterer = OPTICS(
            min_samples=ccfg.optics_min_samples,
            xi=ccfg.optics_xi,
            min_cluster_size=ccfg.optics_min_cluster_size,
        )
        labels = clusterer.fit_predict(Z)
        info["optics_params"] = {
            "min_samples": ccfg.optics_min_samples,
            "xi": ccfg.optics_xi,
            "min_cluster_size": ccfg.optics_min_cluster_size,
        }
        return labels, info

    if method == "dbscan":
        clusterer = DBSCAN(eps=ccfg.dbscan_eps, min_samples=ccfg.dbscan_min_samples)
        labels = clusterer.fit_predict(Z)
        info["dbscan_params"] = {"eps": ccfg.dbscan_eps, "min_samples": ccfg.dbscan_min_samples}
        return labels, info

    raise ValueError(f"Unknown density_method: {ccfg.density_method}")


def hierarchical_cluster(Z: np.ndarray, ccfg: ClusterConfig) -> np.ndarray:
    clusterer = AgglomerativeClustering(n_clusters=ccfg.hier_n_clusters, linkage=ccfg.hier_linkage)
    return clusterer.fit_predict(Z)


# ----------------------------
# Reporting & Visualization
# ----------------------------
def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def plot_embedding(Z2: np.ndarray, labels: np.ndarray, outpath: str, title: str) -> None:
    plt.figure(figsize=(10, 7))
    # Noise label is typically -1
    uniq = np.unique(labels)
    for lb in uniq:
        m = labels == lb
        plt.scatter(Z2[m, 0], Z2[m, 1], s=10, alpha=0.75, label=str(lb))
    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.legend(markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def medoid_index(Z: np.ndarray, idxs: np.ndarray) -> int:
    """
    Medoid in embedding space: point with minimal average distance to others (O(n^2) in cluster size).
    For large clusters, we subsample.
    """
    ids = np.asarray(idxs)
    if len(ids) == 1:
        return int(ids[0])
    # subsample for speed
    max_n = 600
    if len(ids) > max_n:
        rng = np.random.default_rng(42)
        ids_sub = rng.choice(ids, size=max_n, replace=False)
    else:
        ids_sub = ids

    V = Z[ids_sub]
    # compute pairwise distances efficiently
    # (v - w)^2 = v^2 + w^2 - 2vw
    G = V @ V.T
    sq = np.sum(V**2, axis=1, keepdims=True)
    D2 = np.maximum(sq + sq.T - 2.0 * G, 0.0)
    D = np.sqrt(D2)
    score = np.mean(D, axis=1)
    best = ids_sub[np.argmin(score)]
    return int(best)


def plot_cluster_waveforms(
    signals: List[np.ndarray],
    labels: np.ndarray,
    Z: np.ndarray,
    fs: float,
    outdir: str,
    max_clusters: int = 20,
    samples_per_cluster: int = 8,
) -> None:
    """
    For each cluster:
      - plot medoid waveform
      - overlay a few random waveforms (aligned by max-abs peak)
    """
    ensure_dir(outdir)
    rng = np.random.default_rng(42)

    cluster_ids = [c for c in np.unique(labels) if c != -1]
    cluster_ids = sorted(cluster_ids, key=lambda c: np.sum(labels == c), reverse=True)[:max_clusters]

    for c in cluster_ids:
        idxs = np.where(labels == c)[0]
        medoid = medoid_index(Z, idxs)
        pick = idxs if len(idxs) <= samples_per_cluster else rng.choice(idxs, size=samples_per_cluster, replace=False)

        plt.figure(figsize=(10, 5))
        # overlay
        for i in pick:
            x = signals[int(i)]
            if len(x) < 2:
                continue
            # align by peak
            p = int(np.argmax(np.abs(x)))
            # shift time axis so peak at 0
            t = (np.arange(len(x)) - p) / fs * 1e3  # ms
            plt.plot(t, x, alpha=0.25, linewidth=1)

        # medoid highlight
        xm = signals[int(medoid)]
        pm = int(np.argmax(np.abs(xm)))
        tm = (np.arange(len(xm)) - pm) / fs * 1e3
        plt.plot(tm, xm, linewidth=2)

        plt.title(f"Cluster {c} (n={len(idxs)}): waveforms (aligned by peak, ms)")
        plt.xlabel("Time (ms, peak aligned)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"cluster_{c}_waveforms.png"), dpi=160)
        plt.close()


def plot_cluster_psd(
    signals: List[np.ndarray],
    labels: np.ndarray,
    fs: float,
    outdir: str,
    welch_nperseg: int = 1024,
    max_clusters: int = 20,
    max_samples: int = 200,
) -> None:
    """
    Plot mean PSD per cluster (Welch).
    For speed, each cluster uses up to max_samples signals.
    """
    ensure_dir(outdir)
    rng = np.random.default_rng(42)

    cluster_ids = [c for c in np.unique(labels) if c != -1]
    cluster_ids = sorted(cluster_ids, key=lambda c: np.sum(labels == c), reverse=True)[:max_clusters]

    for c in cluster_ids:
        idxs = np.where(labels == c)[0]
        if len(idxs) > max_samples:
            idxs = rng.choice(idxs, size=max_samples, replace=False)

        psd_list = []
        f_ref = None
        for i in idxs:
            x = np.asarray(signals[int(i)], dtype=np.float64)
            if len(x) < 16:
                continue
            nperseg = min(welch_nperseg, len(x))
            noverlap = min(nperseg // 2, nperseg - 1)
            f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant", scaling="density")
            # Interpolate onto reference grid if needed
            if f_ref is None:
                f_ref = f
                psd_list.append(Pxx)
            else:
                # interpolate to f_ref
                psd_list.append(np.interp(f_ref, f, Pxx))

        if f_ref is None or len(psd_list) == 0:
            continue

        P = np.vstack(psd_list)
        Pm = np.mean(P, axis=0)
        plt.figure(figsize=(10, 5))
        plt.semilogy(f_ref, Pm + 1e-20)
        plt.title(f"Cluster {c} mean PSD (Welch) n={len(psd_list)}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"cluster_{c}_psd.png"), dpi=160)
        plt.close()


def plot_cluster_wavelet_energy(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    outdir: str,
    prefix: str = "norm_w_energy_L",
    max_clusters: int = 20
) -> None:
    """
    Use extracted wavelet energy features to plot mean per cluster.
    """
    ensure_dir(outdir)
    cols = [c for c in feature_df.columns if c.startswith(prefix)]
    if len(cols) == 0:
        return

    cluster_ids = [c for c in np.unique(labels) if c != -1]
    cluster_ids = sorted(cluster_ids, key=lambda c: np.sum(labels == c), reverse=True)[:max_clusters]

    for c in cluster_ids:
        idxs = np.where(labels == c)[0]
        m = feature_df.iloc[idxs][cols].mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.bar(np.arange(len(cols)), m.values)
        plt.xticks(np.arange(len(cols)), cols, rotation=45, ha="right")
        plt.title(f"Cluster {c} mean wavelet detail energies (normalized)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"cluster_{c}_wavelet_energy.png"), dpi=160)
        plt.close()


def cluster_summary_table(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    out_csv: str
) -> pd.DataFrame:
    """
    Basic summary: size + key numeric stats (mean of a few informative features).
    """
    df = feature_df.copy()
    df["cluster"] = labels
    # pick a small set of readable features
    key_cols = [c for c in df.columns if c in [
        "norm_env_peak_count",
        "norm_env_top1_top2_dt_ms",
        "norm_spec_centroid",
        "norm_spec_entropy",
        "norm_w_energy_entropy",
        "raw_rms",
        "raw_ptp",
    ]]
    if len(key_cols) == 0:
        # fallback: take first 10 numeric
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        key_cols = [c for c in num_cols if c != "cluster"][:10]

    summary = df.groupby("cluster")[key_cols].agg(["count", "mean", "std"])
    summary.to_csv(out_csv)
    return summary


def feature_anova_report(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    out_csv: str,
    top_k: int = 30
) -> None:
    """
    Quick-and-dirty "which features separate clusters" using ANOVA F-score.
    Ignore noise label (-1).
    """
    m = labels != -1
    y = labels[m]
    X2 = X[m]
    if len(np.unique(y)) < 2:
        return

    F, p = f_classif(X2, y)
    df = pd.DataFrame({"feature": feature_names, "F": F, "p": p})
    df = df.replace([np.inf, -np.inf], np.nan).dropna().sort_values("F", ascending=False)
    df.head(top_k).to_csv(out_csv, index=False)


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="parquet path (or you can modify code to load your own list)")
    parser.add_argument("--signal_col", type=str, default="signal")
    parser.add_argument("--id_col", type=str, default=None)

    parser.add_argument("--fs", type=float, required=True, help="sampling rate Hz")
    parser.add_argument("--outdir", type=str, default="cluster_out")

    parser.add_argument("--int_to_volt", action="store_true")
    parser.add_argument("--scale", type=float, default=1.0, help="int_to_volt scaling factor")

    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--density_method", type=str, default="hdbscan", choices=["hdbscan", "optics", "dbscan"])
    parser.add_argument("--reducer", type=str, default="pca", choices=["pca", "umap"])
    parser.add_argument("--do_hierarchical", action="store_true")
    parser.add_argument("--hier_n_clusters", type=int, default=10)
    args = parser.parse_args()

    ensure_dir(args.outdir)

    fcfg = FeatureConfig(fs=args.fs)
    ccfg = ClusterConfig(
        reducer=args.reducer,
        density_method=args.density_method,
        do_hierarchical=args.do_hierarchical,
        hier_n_clusters=args.hier_n_clusters,
    )

    # Save config
    with open(os.path.join(args.outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_config": asdict(fcfg), "cluster_config": asdict(ccfg)}, f, ensure_ascii=False, indent=2)

    # Load
    signals, ids = load_signals_from_parquet(
        args.input,
        signal_col=args.signal_col,
        id_col=args.id_col,
        int_to_volt=args.int_to_volt,
        scale=args.scale,
    )
    print(f"[INFO] loaded {len(signals)} signals")

    # Feature extraction
    feat_df = extract_features(signals, fcfg, n_jobs=args.n_jobs)
    feat_df.insert(0, "id", ids)
    feat_path = os.path.join(args.outdir, "features.parquet")
    feat_df.to_parquet(feat_path, index=False)
    print(f"[INFO] features saved: {feat_path}")

    # Prepare X
    feature_cols = [c for c in feat_df.columns if c != "id"]
    Xraw = feat_df[feature_cols].values

    # Impute + scale (RobustScaler is more stable with heavy tails)
    imputer = SimpleImputer(strategy="median")
    Ximp = imputer.fit_transform(Xraw)

    scaler = RobustScaler(with_centering=True, with_scaling=True)
    X = scaler.fit_transform(Ximp)

    # Reduce for clustering
    Z, red_info = reduce_features(X, ccfg, random_state=42)
    with open(os.path.join(args.outdir, "reducer_info.json"), "w", encoding="utf-8") as f:
        json.dump(red_info, f, ensure_ascii=False, indent=2)

    # Density clustering
    labels, cl_info = density_cluster(Z, ccfg)
    with open(os.path.join(args.outdir, "cluster_info.json"), "w", encoding="utf-8") as f:
        json.dump(cl_info, f, ensure_ascii=False, indent=2)

    # Optional hierarchical clustering (on reduced space)
    if ccfg.do_hierarchical:
        labels_h = hierarchical_cluster(Z, ccfg)
        feat_df["cluster_hier"] = labels_h

    feat_df["cluster"] = labels
    out_assign = os.path.join(args.outdir, "cluster_assignments.parquet")
    feat_df[["id", "cluster"] + (["cluster_hier"] if ccfg.do_hierarchical else [])].to_parquet(out_assign, index=False)
    print(f"[INFO] cluster assignments saved: {out_assign}")

    # Quick metrics
    n_noise = int(np.sum(labels == -1))
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    print(f"[INFO] clusters={n_clusters}, noise={n_noise}/{len(labels)}")

    # Silhouette (ignore noise)
    m = labels != -1
    if np.sum(m) > 10 and len(np.unique(labels[m])) >= 2:
        try:
            sil = silhouette_score(Z[m], labels[m])
            print(f"[INFO] silhouette (non-noise) = {sil:.4f}")
            with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump({"silhouette_non_noise": float(sil)}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] silhouette failed: {e}")

    # Visualization embedding for plots
    if ccfg.reducer.lower() == "umap" and HAS_UMAP:
        # Use UMAP(2D) directly for viz
        reducer2 = umap.UMAP(
            n_neighbors=ccfg.umap_n_neighbors,
            min_dist=ccfg.umap_min_dist,
            n_components=2,
            random_state=42,
        )
        Z2 = reducer2.fit_transform(X)
        title = "UMAP(2D) embedding colored by cluster"
    else:
        # PCA 2D for viz
        Z2 = pca_2d_for_viz(X)
        title = "PCA(2D) embedding colored by cluster"

    plot_embedding(Z2, labels, os.path.join(args.outdir, "embedding_clusters.png"), title)

    # Per-cluster plots
    plot_cluster_waveforms(
        signals=signals, labels=labels, Z=Z, fs=args.fs,
        outdir=os.path.join(args.outdir, "clusters_waveforms"),
        max_clusters=20, samples_per_cluster=10
    )
    plot_cluster_psd(
        signals=signals, labels=labels, fs=args.fs,
        outdir=os.path.join(args.outdir, "clusters_psd"),
        welch_nperseg=fcfg.welch_nperseg, max_clusters=20, max_samples=250
    )
    plot_cluster_wavelet_energy(
        feature_df=feat_df[feature_cols], labels=labels,
        outdir=os.path.join(args.outdir, "clusters_wavelet_energy"),
        prefix="norm_w_energy_L", max_clusters=20
    )

    # Summary table
    summary = cluster_summary_table(
        feature_df=feat_df[feature_cols],
        labels=labels,
        out_csv=os.path.join(args.outdir, "cluster_summary.csv")
    )
    print(f"[INFO] cluster summary saved: {os.path.join(args.outdir, 'cluster_summary.csv')}")

    # Feature separation report (ANOVA F-score on scaled+imputed original feature space)
    feature_anova_report(
        X=X, labels=labels, feature_names=feature_cols,
        out_csv=os.path.join(args.outdir, "feature_anova_top.csv"),
        top_k=40
    )

    print("[DONE] Results written to:", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
