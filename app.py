"""ECG heartbeat classifier - Streamlit demo.  Research / education only, not a medical device."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as Fn

import ecg_pipeline as ep

ART = Path(__file__).parent / "deploy"
CLASS_COLORS = {"N": "#2e9e5b", "S": "#f28c28", "V": "#d62728", "F": "#7b3fbf"}
CLASS_INFO = {
    "N": "Normal-type beat (includes normal sinus, bundle-branch-block and escape beats).",
    "S": "Supraventricular ectopic beat: an early beat that starts in the atria or AV node.",
    "V": "Ventricular ectopic beat (premature ventricular contraction): a wide, unusually shaped QRS.",
    "F": "Fusion beat: a ventricular and a normal beat merging. Very rare in the training data.",
}

st.set_page_config(page_title="ECG Heartbeat Classifier", page_icon="🫀", layout="wide")


# ----------------------------------------------------------------------------------------
# Model (identical to the training notebook so the saved weights load)
# ----------------------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        if cin == cout and stride == 1:
            self.short = nn.Identity()
        else:
            self.short = nn.Sequential(
                nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm1d(cout),
            )

    def forward(self, x):
        out = Fn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return Fn.relu(out + self.short(x))


class ECGNet(nn.Module):
    def __init__(self, num_classes=4, n_rr=4, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 32, 7, padding=3, bias=False),
                                  nn.BatchNorm1d(32), nn.ReLU())
        self.blocks = nn.Sequential(ResBlock(32, 64, 2), ResBlock(64, 128, 2), ResBlock(128, 128, 2))
        self.rr_branch = nn.Sequential(nn.Linear(n_rr, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128 * 2 + 32, 128), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x, rr):
        h = self.blocks(self.stem(x))
        h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(torch.cat([h, self.rr_branch(rr)], dim=1))


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    cfg = json.load(open(ART / "config.json"))
    pre = np.load(ART / "preprocessing.npz")
    model = ECGNet(len(cfg["class_names"]))
    model.load_state_dict(torch.load(ART / "ecgnet_weights.pth", map_location="cpu"))
    model.eval()
    demo = np.load(ART / "demo_samples.npz")
    demo = {k: demo[k] for k in demo.files}
    return model, cfg, pre["rr_mean"], pre["rr_std"], pre["bias"], demo


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@torch.no_grad()
def predict(model, X, rr_scaled, bias, batch=512):
    """X: (n, 216) normalised beats, rr_scaled: (n, 4). Returns (n, 4) scores."""
    if len(X) == 0:
        return np.empty((0, len(bias)), np.float32)
    outs = []
    for i in range(0, len(X), batch):
        x = torch.from_numpy(np.ascontiguousarray(X[i:i + batch, None, :])).float()
        r = torch.from_numpy(np.ascontiguousarray(rr_scaled[i:i + batch])).float()
        outs.append(model(x, r).numpy())
    return softmax(np.concatenate(outs) + bias)


def gradcam(model, wave, rr_scaled, bias):
    """Grad-CAM on the last residual block (waveform branch only) for the predicted class."""
    x = torch.as_tensor(np.asarray(wave, dtype=np.float32)).reshape(1, 1, -1)
    rr = torch.as_tensor(np.asarray(rr_scaled, dtype=np.float32)).reshape(1, -1)
    store = {}

    def hook(module, inputs, output):
        store["act"] = output
        output.register_hook(lambda g: store.__setitem__("grad", g))

    handle = model.blocks[-1].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            logits = model(x, rr)
            shifted = logits + torch.as_tensor(bias, dtype=torch.float32)
            probs = torch.softmax(shifted, dim=1)[0].detach().numpy()
            pred = int(shifted.argmax(dim=1).item())
            logits[0, pred].backward()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
    act, grad = store["act"].detach(), store["grad"].detach()
    cam = torch.relu((grad.mean(dim=2, keepdim=True) * act).sum(dim=1, keepdim=True))
    cam = Fn.interpolate(cam, size=x.shape[-1], mode="linear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.numpy(), pred, probs


# ----------------------------------------------------------------------------------------
# Display helpers
# ----------------------------------------------------------------------------------------
def explain(pred_name, rr_raw, truth_name=None):
    pre, post, pre_ratio, _ = [float(v) for v in rr_raw]
    if pre_ratio < 0.85:
        timing = "arrived earlier than the recent rhythm would predict"
    elif pre_ratio > 1.15:
        timing = "arrived later than the recent rhythm would predict"
    else:
        timing = "arrived close to the expected time"
    text = (f"**Prediction: {pred_name}.** {CLASS_INFO[pred_name]}\n\n"
            f"The beat came {pre:.2f} s after the previous one and {post:.2f} s before the next, "
            f"so it {timing}. The model combines this timing with the waveform shape.")
    if truth_name is not None:
        text += f"\n\nExpert annotation for this beat: **{truth_name}**" + \
                (" (the model agrees)." if truth_name == pred_name else " (the model disagrees).")
    return text


def plot_wave(signal, cam=None, title=""):
    t = (np.arange(len(signal)) - ep.BEFORE) / ep.FS * 1000
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(t, signal, color="black", linewidth=1.0)
    if cam is not None:
        sc = ax.scatter(t, signal, c=cam, cmap="jet", s=12, vmin=0, vmax=1)
        fig.colorbar(sc, ax=ax, label="Grad-CAM importance")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time from R-peak (ms)")
    ax.set_ylabel("Normalised amplitude")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def show_beat(model, cfg, rr_mean, rr_std, bias, wave, rr_raw, truth_name=None):
    names = cfg["class_names"]
    rr_scaled = (np.asarray(rr_raw, dtype=np.float32) - rr_mean) / rr_std
    cam, pred, probs = gradcam(model, wave, rr_scaled, bias)
    pred_name = names[pred]

    c1, c2, c3 = st.columns([1, 1, 1])
    c1.metric("Predicted class", pred_name)
    c2.metric("Model score", f"{probs[pred] * 100:.1f}%")
    if truth_name is not None:
        c3.metric("Expert annotation", truth_name)
    if pred_name == "F":
        st.warning(f"The model's F predictions are unreliable (test F1 about "
                   f"{cfg['test_metrics'].get('F1_F', 0):.2f}). Treat this as a guess.")

    left, right = st.columns(2)
    with left:
        st.markdown("**ECG heartbeat**")
        fig = plot_wave(wave, title="Normalised beat, R-peak at 0 ms")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Grad-CAM: where the model looked**")
        fig = plot_wave(wave, cam, title=f"Evidence for class {pred_name}")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Class scores**")
    st.bar_chart(pd.DataFrame({"Score": probs}, index=names))
    st.caption("Scores are model outputs after calibration offsets. They are not clinical probabilities.")
    st.markdown(explain(pred_name, rr_raw, truth_name))
    st.caption("Grad-CAM explains the waveform branch only. The RR timing features are used "
               "separately and are not shown in the map.")


def read_csv_signal(file):
    df = pd.read_csv(file)
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        file.seek(0)
        df = pd.read_csv(file, header=None)
        num = df.select_dtypes(include=[np.number])
    return num


# ----------------------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------------------
st.title("🫀 ECG Heartbeat Arrhythmia Classifier")
st.caption("1D ResNet + RR features trained on MIT-BIH (inter-patient split). "
           "Research and education only. This is not a medical device and must not be used for diagnosis.")

if not (ART / "ecgnet_weights.pth").exists():
    st.error("Model files not found. Run the 'Phase 1' cell in the notebook and put the 'deploy' folder "
             "next to app.py.")
    st.stop()

model, cfg, rr_mean, rr_std, bias, demo = load_artifacts()
names = cfg["class_names"]

with st.sidebar:
    st.header("Input")
    mode = st.radio("Mode", ["Demo beats", "Upload a recording (CSV)"])
    method_label = st.selectbox("R-peak detector (upload mode)",
                                ["Auto (WFDB XQRS, fallback built-in)", "Built-in detector"])
    method = "auto" if method_label.startswith("Auto") else "builtin"

with st.expander("Model performance and limitations", expanded=False):
    tm = cfg["test_metrics"]
    st.markdown(
        "Measured on **unseen patients** (MIT-BIH DS2, 49,661 beats):\n"
        f"- Accuracy {tm.get('acc', 0) * 100:.1f}%, macro-F1 {tm.get('macroF1', 0):.2f}\n"
        f"- F1 per class: N {tm.get('F1_N', 0):.2f}, S {tm.get('F1_S', 0):.2f}, "
        f"V {tm.get('F1_V', 0):.2f}, **F {tm.get('F1_F', 0):.2f}**\n\n"
        "Limitations:\n"
        "- **Class F (fusion) is not reliable.** Almost all training F beats come from one patient.\n"
        "- S beats are only moderately detected.\n"
        "- Trained on one lead (MLII) at 360 Hz from one database; other leads, devices or noisy signals "
        "may give wrong results.\n"
        "- In upload mode, R-peaks are detected automatically, which can introduce errors.\n"
        "- Not a clinical tool.")

# ---------------------------------- Demo mode --------------------------------------------
if mode == "Demo beats":
    st.subheader("Demo beats from unseen test patients")
    col_a, col_b = st.columns(2)
    cls_filter = col_a.selectbox("Show beats whose annotation is", ["All"] + names)
    if cls_filter == "All":
        idxs = np.arange(len(demo["label"]))
    else:
        idxs = np.where(demo["label"] == names.index(cls_filter))[0]
    pos = col_b.number_input(f"Beat number (1 to {len(idxs)})", min_value=1, max_value=len(idxs),
                             value=1, step=1)
    i = int(idxs[int(pos) - 1])
    st.caption(f"Source: MIT-BIH record {demo['record'][i]}. The demo set holds about equal numbers of "
               "each class, so it is not representative of real class frequencies.")
    show_beat(model, cfg, rr_mean, rr_std, bias, demo["waveform"][i], demo["rr_raw"][i],
              truth_name=names[int(demo["label"][i])])

# ---------------------------------- Upload mode ------------------------------------------
else:
    st.subheader("Upload a single-lead ECG recording")
    st.markdown(f"CSV with one signal column, **at least {ep.MIN_SECONDS} seconds** long, similar to lead II "
                "(R-peaks pointing up). Only the first "
                f"{ep.MAX_SECONDS} seconds are analysed.")
    up = st.file_uploader("CSV file", type=["csv", "txt"])
    fs = st.number_input("Sampling rate (Hz)", min_value=100.0, max_value=2000.0, value=360.0, step=1.0)
    if up is None:
        st.info("Upload a CSV to start, or switch to 'Demo beats' in the sidebar.")
        st.stop()

    try:
        num = read_csv_signal(up)
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        st.stop()
    if num.shape[1] == 0:
        st.error("No numeric column found in the file.")
        st.stop()
    col = st.selectbox("Signal column (do not pick a time column)", list(num.columns))
    sig = num[col].to_numpy(dtype=float)

    try:
        with st.spinner("Detecting heartbeats..."):
            res = ep.process_recording(sig, fs, method)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    if len(res["X"]) == 0:
        st.error("No usable heartbeats were found. Check the sampling rate, the column and the signal quality.")
        st.stop()

    rr_scaled_all = (res["RR"] - rr_mean) / rr_std
    probs_all = predict(model, res["X"], rr_scaled_all, bias)
    pred_all = probs_all.argmax(1)
    pred_names = np.array([names[k] for k in pred_all])

    peaks_t = res["peaks"] / ep.FS
    hr = 60.0 / np.median(np.diff(peaks_t)) if len(peaks_t) > 2 else float("nan")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Duration", f"{res['duration']:.1f} s")
    m2.metric("Beats classified", len(res["X"]))
    m3.metric("Median heart rate", f"{hr:.0f} bpm")
    m4.metric("Detector used", res["method"])

    counts = pd.Series(pred_names).value_counts().reindex(names, fill_value=0)
    st.markdown("**Predicted class counts**")
    st.dataframe(pd.DataFrame({"Beats": counts, "Share": (counts / counts.sum() * 100).round(1).astype(str) + " %"}))
    st.caption("Check the detected R-peaks below. If they are wrong, results are not meaningful.")

    win = 10.0
    max_start = max(0.0, res["duration"] - win)
    start = st.slider("View window start (s)", 0.0, float(max_start), 0.0, 1.0) if max_start > 0 else 0.0
    a, b = int(start * ep.FS), int((start + win) * ep.FS)
    fig, ax = plt.subplots(figsize=(13, 3.2))
    tt = np.arange(a, min(b, len(res["signal"]))) / ep.FS
    ax.plot(tt, res["signal"][a:b], color="black", linewidth=0.8)
    for name in names:
        sel = (pred_names == name) & (res["beat_peaks"] >= a) & (res["beat_peaks"] < b)
        if sel.any():
            pk = res["beat_peaks"][sel]
            ax.scatter(pk / ep.FS, res["signal"][pk], s=40, color=CLASS_COLORS.get(name, "gray"),
                       label=name, zorder=3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalised amplitude")
    ax.grid(alpha=0.3)
    ax.legend(title="Predicted", loc="upper right", ncol=4)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Inspect one beat")
    labels = [f"#{k + 1} at {res['beat_peaks'][k] / ep.FS:.2f} s: {pred_names[k]}" for k in range(len(pred_names))]
    choice = st.selectbox("Beat", range(len(labels)), format_func=lambda k: labels[k])
    show_beat(model, cfg, rr_mean, rr_std, bias, res["X"][choice], res["RR"][choice])
