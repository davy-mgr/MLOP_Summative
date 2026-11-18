"""Streamlit dashboard: predict, visualise, upload + retrain, monitor."""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
CLASSES = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
           "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

st.set_page_config(page_title="UrbanSound8K Classifier", layout="wide")


def api(method, path, timeout=120, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=timeout, **kw)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        st.error(f"{e.response.status_code}: {e.response.text}")
    except requests.RequestException as e:
        st.error(f"API unreachable at {API}: {e}")
    return None


def fmt_uptime(s):
    s = int(s)
    return f"{s // 3600}h {s % 3600 // 60}m {s % 60}s"


page = st.sidebar.radio("Page", ["Predict", "Visualizations", "Upload & Retrain", "Monitoring"])
st.sidebar.caption(f"API: {API}")

# ------------------------------------------------------------------ Predict
if page == "Predict":
    st.title("Urban sound classification")
    f = st.file_uploader("Upload a .wav clip (up to ~4 s)", type=["wav"])
    if f:
        st.audio(f.getvalue(), format="audio/wav")
        if st.button("Predict", type="primary"):
            res = api("POST", "/predict", params={"spectrogram": "true"},
                      files={"file": (f.name, f.getvalue(), "audio/wav")})
            if res:
                c1, c2 = st.columns(2)
                c1.metric("Predicted class", res["class"].replace("_", " "))
                c1.metric("Confidence", f"{res['confidence']:.1%}")
                c1.caption(f"Inference latency: {res['latency_ms']} ms")
                c1.bar_chart(pd.Series(res["probabilities"]).sort_values())
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.imshow(np.array(res["spectrogram"]), origin="lower", aspect="auto", cmap="magma")
                ax.set(title="Log-mel spectrogram", xlabel="frames", ylabel="mel bin")
                c2.pyplot(fig)

# ------------------------------------------------------------------ Visualizations
elif page == "Visualizations":
    st.title("What does the data look like?")
    dist = api("GET", "/viz/class-distribution")
    if dist:
        df = pd.DataFrame(dist).rename_axis("class")
        st.subheader("1. Class balance")
        st.bar_chart(df[["train", "test", "uploads"]])
        st.caption("Check how balanced the classes are. Under-represented classes (the UrbanSound8K "
                   "car_horn and gun_shot classes have far fewer clips than the rest) mean overall "
                   "accuracy can hide poor recall on those classes, so we also report macro F1.")
    with st.spinner("Computing spectral features (first run only)..."):
        feats = api("GET", "/viz/features", timeout=600)
        specs = api("GET", "/viz/mean-spectrograms", timeout=600)
    if feats:
        st.subheader("2. Spectral centroid, zero-crossing rate and RMS energy by class")
        cols = st.columns(3)
        titles = {"spectral_centroid": "Spectral centroid (Hz)",
                  "zero_crossing_rate": "Zero-crossing rate", "rms": "RMS energy"}
        for col, key in zip(cols, titles):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.boxplot([feats[c][key] for c in feats])
            ax.set_xticklabels(list(feats), rotation=90, fontsize=8)
            ax.set_title(titles[key])
            col.pyplot(fig)
        st.caption("Centroid and ZCR track how 'bright' or noisy a sound is; RMS tracks loudness. "
                   "Classes whose boxes barely overlap are easy to separate even with simple features; "
                   "heavily overlapping classes are where the CNN has to work harder.")
    if specs:
        st.subheader("3. Average mel-spectrogram per class")
        fig, axes = plt.subplots(2, 5, figsize=(18, 6))
        for ax, c in zip(axes.ravel(), specs):
            ax.imshow(np.array(specs[c]), origin="lower", aspect="auto", cmap="magma")
            ax.set_title(c.replace("_", " "), fontsize=9)
            ax.axis("off")
        st.pyplot(fig)
        st.caption("Tonal/harmonic sounds (siren, car horn) show horizontal bands, impulsive sounds "
                   "(gun shot) show a short vertical burst, and steady machinery (air conditioner, "
                   "engine idling) shows a smooth low-frequency-heavy texture.")

# ------------------------------------------------------------------ Upload & Retrain
elif page == "Upload & Retrain":
    st.title("Upload new data and retrain")
    st.markdown("Upload **many .wav files with a class label**, or a **.zip** whose sub-folders are named "
                "after the classes (`dog_bark/*.wav`, `siren/*.wav`, ...).")
    label = st.selectbox("Label for the .wav files (ignored for zip folders)", CLASSES)
    files = st.file_uploader("Files", type=["wav", "zip"], accept_multiple_files=True)
    if files and st.button("Upload"):
        res = api("POST", "/upload", data={"label": label},
                  files=[("files", (x.name, x.getvalue())) for x in files], timeout=600)
        if res:
            st.success(f"Saved {res['saved']} file(s). {res['pending_uploads']} waiting for retraining.")
            if res["rejected"]:
                st.warning(f"Rejected: {res['rejected'][:10]}")
            if res["auto_retrain_started"]:
                st.info("Automatic retraining was triggered (threshold reached).")

    st.divider()
    s = api("GET", "/status")
    if s:
        st.metric("Files waiting", s["pending_uploads"],
                  help=f"Retraining starts automatically at {s['auto_retrain_threshold']} files")
        c1, c2, c3 = st.columns(3)
        epochs = c1.slider("Epochs", 1, 50, 15)
        scratch = c2.checkbox("Retrain from scratch (default: fine-tune)")
        if c3.button("Retrain model", type="primary"):
            api("POST", "/retrain", params={"epochs": epochs, "from_scratch": scratch})
        if st.button("Train from scratch on the original dataset"):
            api("POST", "/train", params={"epochs": 40})

        j = api("GET", "/retrain/status")
        if j and (j["running"] or j["message"] != "idle"):
            if j["running"]:
                st.progress(min(j["epoch"] / max(j["epochs"], 1), 1.0), text=j["message"])
                time.sleep(3)
                st.rerun()
            elif j["error"]:
                st.error(j["error"])
            else:
                r = j["result"]
                st.success(f"Finished: model v{r['version']} - accuracy {r['accuracy']:.3f}, "
                           f"macro-F1 {r['f1_macro']:.3f}, "
                           f"{'promoted to live model' if r['promoted'] else 'NOT promoted (worse than current)'}")

# ------------------------------------------------------------------ Monitoring
else:
    st.title("Model monitoring")
    s = api("GET", "/status")
    if s:
        c = st.columns(4)
        c[0].metric("API uptime", fmt_uptime(s["uptime_seconds"]))
        c[1].metric("Model version", s["model_version"] or "none")
        c[2].metric("Predictions served", s["predictions_served"])
        c[3].metric("Errors", s["errors"])
        c = st.columns(4)
        c[0].metric("Avg latency (ms)", s["latency_ms_avg"] or "-")
        c[1].metric("p95 latency (ms)", s["latency_ms_p95"] or "-")
        c[2].metric("Test accuracy", f"{s['model_accuracy']:.3f}" if s["model_accuracy"] else "-")
        c[3].metric("Last trained (UTC)", (s["last_trained_at"] or "-")[:19])

    m = api("GET", "/metrics")
    if m and m["current"]:
        cur = m["current"]
        st.subheader(f"Live model v{cur['version']} - evaluation on held-out test fold")
        mm = cur["metrics"]
        c = st.columns(4)
        c[0].metric("Accuracy", f"{mm['accuracy']:.3f}")
        c[1].metric("Macro precision", f"{mm['precision_macro']:.3f}")
        c[2].metric("Macro recall", f"{mm['recall_macro']:.3f}")
        c[3].metric("Macro F1", f"{mm['f1_macro']:.3f}")

        l, r = st.columns(2)
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = np.array(mm["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(10), CLASSES, rotation=90, fontsize=7)
        ax.set_yticks(range(10), CLASSES, fontsize=7)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6)
        ax.set(xlabel="Predicted", ylabel="True", title="Confusion matrix")
        l.pyplot(fig)
        h = pd.DataFrame({k: v for k, v in cur["history"].items()})
        r.line_chart(h[[c for c in ("accuracy", "val_accuracy") if c in h]])
        r.line_chart(h[[c for c in ("loss", "val_loss") if c in h]])
        st.subheader("Model versions")
        st.dataframe(pd.DataFrame(m["versions"]), use_container_width=True)
    if st.button("Refresh"):
        st.rerun()

