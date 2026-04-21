"""
Physiological feature extraction — 19 ANS-relevant features.

Extracts a concise, physiologically-principled feature set from five
modalities (ECG, PPG, Resp, EDA, Skin Temperature) using a sliding
window approach.  Each feature targets a specific autonomic nervous
system (ANS) axis following Kreibig (Biological Psychology, 2010).

Modality → Feature mapping (Table II in paper):
  ECG  : SDNN, RMSSD, HF Power, LF/HF, SampEn, pNN50, Inst HR
  PPG  : Pulse Rate, Pulse Amplitude Variability (PAV), PTT
  Resp : Breath Rate, RVT, Inst Breath Rate
  EDA  : SCL, SCR Freq, Phasic Amp, SCR Rise Time
  Temp : Skin Temp, dSkin Temp

Reference: Section III.C (Sliding Window Feature Extraction),
           Table II, and Supplementary B.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.signal import medfilt, butter, sosfiltfilt
from scipy.ndimage import median_filter


@dataclass
class FeatureResult:
    """Output of the feature extraction pipeline."""
    matrix: np.ndarray              # (n_windows, n_features) — normalized
    labels: list[str]               # feature names
    time_vec: np.ndarray            # window center times (seconds)
    modality_map: dict[str, list[int]]  # modality → column indices
    raw_matrix: np.ndarray          # before normalization


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-modality denoising (Supplementary B)
# ═══════════════════════════════════════════════════════════════════════════════

def _interp_nans(arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN gaps; fill edges with nearest valid."""
    out = arr.copy().astype(float)
    nans = np.isnan(out)
    if nans.any() and not nans.all():
        xi = np.arange(len(out))
        out[nans] = np.interp(xi[nans], xi[~nans], out[~nans])
    elif nans.all():
        out[:] = 0
    return out


def _clip_iqr(sig: np.ndarray, k: float = 5.0) -> np.ndarray:
    """Clip outliers beyond k × IQR from median (Tukey fence)."""
    valid = sig[~np.isnan(sig)]
    if len(valid) < 10:
        return sig
    q1, q3 = np.percentile(valid, [25, 75])
    iqr = q3 - q1
    return np.clip(sig, q1 - k * iqr, q3 + k * iqr)


def _denoise_ecg(ecg: np.ndarray, fs: int) -> np.ndarray:
    """Bandpass 0.5–40 Hz for QRS preservation + baseline removal."""
    sig = _clip_iqr(_interp_nans(ecg), k=6.0)
    nyq = fs / 2
    lo, hi = 0.5 / nyq, min(40.0, nyq * 0.95) / nyq
    sos = butter(3, [lo, hi], btype="band", output="sos")
    return sosfiltfilt(sos, sig)


def _denoise_ppg(ppg: np.ndarray, fs: int) -> np.ndarray:
    """Bandpass 0.4–8 Hz for pulse wave isolation."""
    sig = _clip_iqr(_interp_nans(ppg), k=5.0)
    nyq = fs / 2
    lo, hi = 0.4 / nyq, min(8.0, nyq * 0.95) / nyq
    sos = butter(3, [lo, hi], btype="band", output="sos")
    return sosfiltfilt(sos, sig)


def _denoise_resp(resp: np.ndarray, fs: int) -> np.ndarray:
    """Bandpass 0.05–1 Hz for respiratory band isolation."""
    sig = _clip_iqr(_interp_nans(resp), k=5.0)
    nyq = fs / 2
    lo, hi = 0.05 / nyq, min(1.0, nyq * 0.95) / nyq
    sos = butter(3, [lo, hi], btype="band", output="sos")
    return sosfiltfilt(sos, sig)


def _denoise_eda(eda: np.ndarray, fs: int) -> np.ndarray:
    """Median filter for spikes + lowpass for smoothing."""
    sig = _clip_iqr(_interp_nans(eda), k=4.0)
    kernel = int(0.5 * fs) | 1
    sig = medfilt(sig, kernel_size=max(kernel, 3))
    nyq = fs / 2
    cutoff = min(1.0, nyq * 0.8)
    hi = cutoff / nyq
    if 0 < hi < 1 and len(sig) > 15:
        sos = butter(2, hi, btype="low", output="sos")
        sig = sosfiltfilt(sos, sig)
    return sig


def _denoise_temp(temp: np.ndarray, fs: int) -> np.ndarray:
    """3-second median filter for jump removal."""
    sig = _clip_iqr(_interp_nans(temp), k=4.0)
    kernel = int(3.0 * fs) | 1
    return medfilt(sig, kernel_size=max(kernel, 3))


# ═══════════════════════════════════════════════════════════════════════════════
#  Windowed statistics
# ═══════════════════════════════════════════════════════════════════════════════

def _rolling_mean(sig, fs, win_sec, step_sec, min_valid_frac=0.5):
    """Windowed mean with NaN handling."""
    win = int(win_sec * fs)
    step = int(step_sec * fs)
    min_valid = int(win * min_valid_frac)
    starts = np.arange(0, len(sig) - win + 1, step)
    times = (starts + win / 2) / fs
    values = np.empty(len(starts))
    for j, s in enumerate(starts):
        chunk = sig[s:s + win]
        n_valid = np.sum(~np.isnan(chunk))
        values[j] = np.nanmean(chunk) if n_valid >= min_valid else np.nan
    return times, values


def _rolling_std(sig, fs, win_sec, step_sec, min_valid_frac=0.5):
    """Windowed standard deviation with NaN handling."""
    win = int(win_sec * fs)
    step = int(step_sec * fs)
    min_valid = int(win * min_valid_frac)
    starts = np.arange(0, len(sig) - win + 1, step)
    times = (starts + win / 2) / fs
    values = np.empty(len(starts))
    for j, s in enumerate(starts):
        chunk = sig[s:s + win]
        n_valid = np.sum(~np.isnan(chunk))
        values[j] = np.nanstd(chunk) if n_valid >= min_valid else np.nan
    return times, values


# ═══════════════════════════════════════════════════════════════════════════════
#  Main feature extraction (Algorithm 1, Step 1–2)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features(
    ecg_clean: np.ndarray,
    ppg_clean: np.ndarray,
    resp_clean: np.ndarray,
    eda_clean: np.ndarray,
    temp_clean: np.ndarray,
    ecg_fs: int = 200,
    ppg_fs: int = 200,
    resp_fs: int = 100,
    eda_fs: int = 4,
    temp_fs: int = 4,
    window_sec: int = 30,
    step_sec: int = 1,
) -> FeatureResult:
    """
    Extract 19 physiologically-principled features using a sliding window.

    Default parameters match the paper: 30 s window, 1 s step.

    Parameters
    ----------
    ecg_clean, ppg_clean, resp_clean, eda_clean, temp_clean : 1-D arrays
        Pre-cleaned physiological signals at their native sampling rates.
    ecg_fs, ppg_fs, resp_fs, eda_fs, temp_fs : int
        Sampling rates (Hz) for each modality.
    window_sec : int
        Window length in seconds.
    step_sec : int
        Step size in seconds.

    Returns
    -------
    FeatureResult
        Normalized feature matrix, labels, time vector, and modality map.
    """
    import neurokit2 as nk
    import heartpy as hp
    import flirt

    # ── Denoise ──────────────────────────────────────────────
    ecg_dn = _denoise_ecg(ecg_clean, ecg_fs)
    ppg_dn = _denoise_ppg(ppg_clean, ppg_fs)
    resp_dn = _denoise_resp(resp_clean, resp_fs)
    eda_dn = _denoise_eda(eda_clean, eda_fs)
    temp_dn = _denoise_temp(temp_clean, temp_fs)

    # ── ECG: 7 features ─────────────────────────────────────
    signals_ecg, _ = nk.ecg_peaks(ecg_dn, sampling_rate=ecg_fs,
                                   correct_artifacts=True, show=False)
    r_peaks = np.where(signals_ecg["ECG_R_Peaks"] == 1)[0]
    if len(r_peaks) < 10:
        raise ValueError(f"ECG: only {len(r_peaks)} R-peaks — signal too noisy")

    ibi_ecg = hp.analysis.calc_rr(r_peaks, sample_rate=ecg_fs)
    rr_arr = np.array(ibi_ecg["RR_list"], dtype=float)
    bad_ibi = (rr_arr < 300) | (rr_arr > 2000)
    if bad_ibi.any():
        valid_mask = ~bad_ibi
        if valid_mask.sum() > 2:
            xi = np.arange(len(rr_arr))
            rr_arr[bad_ibi] = np.interp(xi[bad_ibi], xi[valid_mask],
                                         rr_arr[valid_mask])
        else:
            rr_arr[bad_ibi] = np.nanmedian(rr_arr)

    ts_rp = np.arange(len(ecg_dn)) / ecg_fs
    pd_rpeaks = pd.DataFrame({"ibi": rr_arr.tolist()})
    pd_rpeaks.index = pd.to_datetime(
        (ts_rp[r_peaks[:-1]] * 1000).astype(int), unit="ms", utc=True)
    hrv = flirt.get_hrv_features(
        pd_rpeaks["ibi"], window_sec, step_sec,
        ["td", "fd", "stat", "nl"], threshold=0.5)

    ecg_feats = [hrv["hrv_sdnn"].values, hrv["hrv_rmssd"].values,
                 hrv["hrv_hf"].values, hrv["hrv_lf_hf_ratio"].values,
                 hrv["hrv_entropy"].values, hrv["hrv_pnni_50"].values]

    # Instantaneous HR
    inst_hr = 60000.0 / rr_arr
    t_uniform = np.arange(0, ts_rp[-1], 1.0)
    inst_hr_uniform = np.interp(t_uniform, ts_rp[r_peaks[:-1]], inst_hr)
    _, ecg_ihr = _rolling_mean(inst_hr_uniform, 1, window_sec, step_sec)
    ecg_feats.append(ecg_ihr)
    ecg_labels = ["SDNN", "RMSSD", "HF Power", "LF/HF", "SampEn",
                  "pNN50", "Inst HR"]

    # ── PPG: 3 features ──────────────────────────────────────
    sigs_ppg = nk.ppg_findpeaks(ppg_dn, sampling_rate=ppg_fs, show=False)
    ppg_peaks = sigs_ppg["PPG_Peaks"]

    ibi_ppg = hp.analysis.calc_rr(ppg_peaks, sample_rate=ppg_fs)
    ts_ppg = np.arange(len(ppg_dn)) / ppg_fs
    pd_ppg = pd.DataFrame({"ibi": ibi_ppg["RR_list"]})
    pd_ppg.index = pd.to_datetime(
        (ts_ppg[ppg_peaks[:-1]] * 1000).astype(int), unit="ms", utc=True)
    hrv_ppg = flirt.get_hrv_features(
        pd_ppg["ibi"], window_sec, step_sec, ["td"], threshold=0.5)
    ppg_rate = hrv_ppg["hrv_mean_hr"].values

    ppg_amp = np.abs(ppg_dn[ppg_peaks])
    ppg_amp_ts = np.interp(np.arange(len(ppg_dn)) / ppg_fs,
                           ppg_peaks / ppg_fs, ppg_amp)
    _, ppg_pav = _rolling_std(ppg_amp_ts, ppg_fs, window_sec, step_sec)

    # PTT (pulse transit time = R-peak to PPG-peak delay)
    ptt_values, ptt_times = [], []
    for i in range(len(r_peaks) - 1):
        r_idx = r_peaks[i]
        search_start = r_idx + int(0.1 * ecg_fs)
        search_end = min(r_idx + int(0.5 * ecg_fs), r_peaks[i + 1], len(ppg_dn))
        if search_start >= search_end:
            continue
        seg = ppg_dn[search_start:search_end]
        if len(seg) < 3:
            continue
        ppg_peak_idx = search_start + np.argmax(seg)
        ptt_sec = (ppg_peak_idx - r_idx) / ecg_fs
        if 0.1 <= ptt_sec <= 0.5:
            ptt_values.append(ptt_sec * 1000)
            ptt_times.append(ts_rp[r_idx])
    if len(ptt_values) > 10:
        ptt_uniform = np.interp(t_uniform, ptt_times, ptt_values)
        _, ppg_ptt = _rolling_mean(ptt_uniform, 1, window_sec, step_sec)
    else:
        ppg_ptt = np.full(len(ppg_rate), np.nan)

    ppg_feats = [ppg_rate, ppg_pav, ppg_ptt]
    ppg_labels = ["Pulse Rate", "PAV", "PTT"]

    # ── Resp: 3 features ─────────────────────────────────────
    rr = nk.rsp_rate(resp_dn, sampling_rate=resp_fs, method="trough")
    _, resp_rate = _rolling_mean(rr, resp_fs, window_sec, step_sec)

    rvt = nk.rsp_rvt(resp_dn, sampling_rate=resp_fs, method="power2020",
                      show=False)
    rvt_arr = rvt.values.ravel() if hasattr(rvt, "values") else np.asarray(rvt).ravel()
    _, resp_rvt = _rolling_mean(rvt_arr, resp_fs, window_sec, step_sec)

    resp_peaks_dict = nk.rsp_findpeaks(resp_dn, sampling_rate=resp_fs)
    resp_peaks_idx = resp_peaks_dict.get(
        "RSP_Peaks", resp_peaks_dict.get("RSP_Troughs", np.array([])))
    if len(resp_peaks_idx) > 2:
        rp_times = resp_peaks_idx / resp_fs
        inst_br = 60.0 / np.diff(rp_times)
        br_uniform = np.interp(t_uniform, rp_times[:-1], inst_br)
        _, resp_ibr = _rolling_mean(br_uniform, 1, window_sec, step_sec)
    else:
        resp_ibr = resp_rate.copy()

    resp_feats = [resp_rate, resp_rvt, resp_ibr]
    resp_labels = ["Breath Rate", "RVT", "Inst Breath Rate"]

    # ── EDA: 4 features ──────────────────────────────────────
    eda_arr = np.asarray(eda_dn, dtype=float)
    try:
        eda_dec = nk.eda_phasic(eda_arr, sampling_rate=eda_fs, method="cvxeda")
    except Exception:
        try:
            eda_dec = nk.eda_phasic(eda_arr, sampling_rate=eda_fs,
                                     method="highpass")
        except Exception:
            from scipy.ndimage import uniform_filter1d
            kernel = int(4.0 * eda_fs) | 1
            tonic = uniform_filter1d(eda_arr, size=max(kernel, 3))
            eda_dec = pd.DataFrame({"EDA_Tonic": tonic,
                                    "EDA_Phasic": eda_arr - tonic})
    scl = eda_dec["EDA_Tonic"].values
    _, eda_scl = _rolling_mean(scl, eda_fs, window_sec, step_sec)

    scr = eda_dec["EDA_Phasic"].values
    scr_peaks_result = nk.eda_findpeaks(scr, sampling_rate=eda_fs)
    if isinstance(scr_peaks_result, dict):
        scr_peaks_idx = scr_peaks_result.get("SCR_Peaks", np.array([]))
    elif isinstance(scr_peaks_result, tuple):
        scr_peaks_idx = scr_peaks_result[0]
        if isinstance(scr_peaks_idx, dict):
            scr_peaks_idx = scr_peaks_idx.get("SCR_Peaks", np.array([]))
    else:
        scr_peaks_idx = np.array([])

    scr_binary = np.zeros(len(scr))
    valid_peaks = np.asarray(scr_peaks_idx, dtype=int)
    valid_peaks = valid_peaks[valid_peaks < len(scr_binary)]
    if len(valid_peaks) > 0:
        scr_binary[valid_peaks] = 1
    win_s = int(window_sec * eda_fs)
    step_s = int(step_sec * eda_fs)
    starts = np.arange(0, len(scr_binary) - win_s + 1, step_s)
    eda_scr_freq = np.array([np.sum(scr_binary[s:s + win_s]) for s in starts])

    _, eda_phasic_amp = _rolling_mean(np.abs(scr), eda_fs, window_sec, step_sec)

    # SCR rise time
    scr_idx_arr = np.asarray(scr_peaks_idx, dtype=int) if len(scr_peaks_idx) > 0 \
        else np.array([], dtype=int)
    if len(scr_idx_arr) > 2:
        rise_times, rise_positions = [], []
        for pk in scr_idx_arr:
            if pk >= len(scr) or pk < 1:
                continue
            onset = pk
            for j in range(pk - 1, max(pk - int(5 * eda_fs), 0) - 1, -1):
                if scr[j] <= 0 or scr[j] > scr[j + 1]:
                    onset = j
                    break
            rt = (pk - onset) / eda_fs
            if 0.1 <= rt <= 5.0:
                rise_times.append(rt)
                rise_positions.append(pk / eda_fs)
        if len(rise_times) > 3:
            rt_uniform = np.interp(
                np.arange(0, len(eda_arr) / eda_fs, 1.0),
                rise_positions, rise_times)
            _, eda_rise_time = _rolling_mean(rt_uniform, 1, window_sec, step_sec)
        else:
            eda_rise_time = np.full(len(eda_scl), np.nan)
    else:
        eda_rise_time = np.full(len(eda_scl), np.nan)

    eda_feats = [eda_scl, eda_scr_freq, eda_phasic_amp, eda_rise_time]
    eda_labels = ["SCL", "SCR Freq", "Phasic Amp", "SCR Rise Time"]

    # ── Temp: 2 features ─────────────────────────────────────
    _, temp_mean = _rolling_mean(temp_dn, temp_fs, window_sec, step_sec)
    temp_deriv = np.gradient(temp_dn, 1.0 / temp_fs)
    _, temp_dmean = _rolling_mean(temp_deriv, temp_fs, window_sec, step_sec)
    temp_feats = [temp_mean, temp_dmean]
    temp_labels = ["Skin Temp", "dSkin Temp"]

    # ── Assemble & normalize ─────────────────────────────────
    all_feats = ecg_feats + ppg_feats + resp_feats + eda_feats + temp_feats
    all_labels = ecg_labels + ppg_labels + resp_labels + eda_labels + temp_labels
    min_len = min(len(f) for f in all_feats)
    all_feats = [f[:min_len] for f in all_feats]

    t_vec, _ = _rolling_mean(temp_dn, temp_fs, window_sec, step_sec)
    t_vec = t_vec[:min_len]
    raw_matrix = np.column_stack(all_feats)

    # Drop constant or all-NaN columns
    keep = np.ones(raw_matrix.shape[1], dtype=bool)
    for i in range(raw_matrix.shape[1]):
        col = raw_matrix[:, i]
        if np.all(np.isnan(col)) or np.nanstd(col) < 1e-10:
            keep[i] = False
    raw_matrix = raw_matrix[:, keep]
    all_labels = [l for l, k in zip(all_labels, keep) if k]

    # Robust percentile normalization (2nd–98th)
    norm_matrix = np.zeros_like(raw_matrix)
    for i in range(raw_matrix.shape[1]):
        col = raw_matrix[:, i]
        valid = col[~np.isnan(col)]
        if len(valid) < 5:
            continue
        p2, p98 = np.percentile(valid, [2, 98])
        rng = p98 - p2
        if rng > 1e-12:
            norm_matrix[:, i] = np.clip((col - p2) / rng, 0, 1)
        else:
            mn, mx = np.nanmin(col), np.nanmax(col)
            rng2 = mx - mn
            norm_matrix[:, i] = (col - mn) / rng2 if rng2 > 1e-12 else 0.5

    # Drop features with >10% missing, then interpolate remainder
    missing = np.mean(np.isnan(norm_matrix), axis=0)
    keep2 = missing < 0.10
    norm_matrix = norm_matrix[:, keep2]
    raw_matrix = raw_matrix[:, keep2]
    all_labels = [l for l, k in zip(all_labels, keep2) if k]

    for col in range(norm_matrix.shape[1]):
        mask = np.isnan(norm_matrix[:, col])
        if mask.any():
            xi = np.arange(len(norm_matrix))
            good = ~mask
            if good.any():
                norm_matrix[mask, col] = np.interp(xi[mask], xi[good],
                                                    norm_matrix[good, col])
            else:
                norm_matrix[:, col] = 0

    # Build modality map
    mod_map: dict[str, list[int]] = {"ECG": [], "PPG": [], "Resp": [],
                                      "EDA": [], "Temp": []}
    for i, lbl in enumerate(all_labels):
        if lbl in set(ecg_labels):
            mod_map["ECG"].append(i)
        elif lbl in set(ppg_labels):
            mod_map["PPG"].append(i)
        elif lbl in set(resp_labels):
            mod_map["Resp"].append(i)
        elif lbl in set(eda_labels):
            mod_map["EDA"].append(i)
        elif lbl in set(temp_labels):
            mod_map["Temp"].append(i)
    mod_map = {k: v for k, v in mod_map.items() if v}

    return FeatureResult(
        matrix=norm_matrix, labels=all_labels, time_vec=t_vec,
        modality_map=mod_map, raw_matrix=raw_matrix,
    )
