import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import time
import tempfile
import os
import pandas as pd

from gradcam import compute_gradcam, overlay_heatmap
from utils import preprocess_for_model, get_last_conv_layer

st.set_page_config(
    page_title="DeepShield — Media Authentication",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Accent color — hardcoded so theme switching still works freely ─────────────
# We do NOT use var(--primary-color) here because that requires config.toml
# which locks the theme switcher. Instead we hardcode our blue and let
# Streamlit handle all text/background colours via its own theme engine.
BLUE = "#2563eb"

st.markdown(f"""
<style>
/* ── Hide footer ── */
footer {{visibility: hidden;}}

/* ── Tab BAR only — scoped tightly so opacity never leaks into content ── */
.stTabs [data-baseweb="tab-list"] {{
    background: var(--secondary-background-color);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(128,128,128,0.2);
}}
/* inactive tab BUTTONS only */
.stTabs [data-baseweb="tab-list"] [role="tab"] {{
    background: transparent;
    border-radius: 8px;
    color: var(--text-color);
    font-weight: 500;
    padding: 8px 20px;
    border: none;
    opacity: 0.6;
}}
/* active tab BUTTON */
.stTabs [data-baseweb="tab-list"] [aria-selected="true"] {{
    background: {BLUE} !important;
    color: white !important;
    opacity: 1 !important;
}}
/* tab PANEL — always fully visible, never inherits opacity */
[role="tabpanel"],
[role="tabpanel"] * {{
    opacity: 1 !important;
}}

/* ── Metric cards ── */
[data-testid="stMetric"] {{
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 12px;
    padding: 16px;
}}

/* ── Primary buttons ── */
.stButton > button {{
    background: {BLUE};
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 0.95rem;
    width: 100%;
    transition: opacity 0.2s ease;
}}
.stButton > button:hover {{
    opacity: 0.85;
}}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {{
    background: var(--secondary-background-color);
    color: {BLUE} !important;
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 8px;
    font-weight: 500;
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background: var(--secondary-background-color);
    border: 2px dashed rgba(128,128,128,0.3);
    border-radius: 12px;
    padding: 8px;
}}

/* ── Sidebar toggle always visible ── */
[data-testid="stSidebarCollapsedControl"] {{
    background: var(--secondary-background-color) !important;
    border: 1px solid rgba(128,128,128,0.3) !important;
    border-radius: 0 8px 8px 0 !important;
}}

/* ── Progress bar ── */
.stProgress > div > div {{
    background: {BLUE};
    border-radius: 8px;
}}
</style>
""", unsafe_allow_html=True)


# ── Hero header ────────────────────────────────────────────────────────────────
def render_header():
    st.markdown(f"""
    <div style="
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
    ">
        <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
            <div style="font-size:2.6rem;">🛡️</div>
            <div>
                <div style="
                    font-size:1.9rem;
                    font-weight:800;
                    color:var(--text-color);
                    letter-spacing:-0.02em;
                    line-height:1.1;
                ">DeepShield</div>
                <div style="
                    font-size:0.93rem;
                    color:{BLUE};
                    font-weight:500;
                    margin-top:4px;
                ">AI-Powered Media Authentication System</div>
            </div>
            <div style="margin-left:auto; text-align:right;">
                <span style="
                    background:rgba(37,99,235,0.12);
                    border:1px solid {BLUE};
                    color:{BLUE};
                    padding:4px 12px;
                    border-radius:20px;
                    font-size:0.78rem;
                    font-weight:600;
                ">EfficientNetB0 + Grad-CAM</span>
                <div style="
                    color:var(--text-color);
                    opacity:0.5;
                    font-size:0.74rem;
                    margin-top:8px;
                ">CDS6334 Visual Information Processing</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Verdict card ───────────────────────────────────────────────────────────────
def render_verdict(is_fake, fake_prob, real_prob, inference_ms, heatmap=None):
    if is_fake:
        color      = "#ef4444"
        bg         = "rgba(239,68,68,0.10)"
        icon       = "⚠️"
        verdict    = "DEEPFAKE DETECTED"
        conf_label = "Fake"
        conf_value = fake_prob
    else:
        color      = "#22c55e"
        bg         = "rgba(34,197,94,0.10)"
        icon       = "✅"
        verdict    = "AUTHENTIC"
        conf_label = "Real"
        conf_value = real_prob

    st.markdown(f"""
    <div style="
        background:{bg};
        border:2px solid {color};
        border-radius:14px;
        padding:20px 24px;
        margin-bottom:16px;
        text-align:center;
    ">
        <div style="font-size:2rem; margin-bottom:6px;">{icon}</div>
        <div style="
            font-size:1.3rem;
            font-weight:800;
            color:{color};
            letter-spacing:0.05em;
        ">{verdict}</div>
        <div style="
            font-size:0.8rem;
            color:var(--text-color);
            margin-top:4px;
        ">Confidence: {conf_value*100:.1f}%</div>
    </div>

    <div style="color:var(--text-color); font-size:0.8rem; margin:8px 0 4px 0;">
        {conf_label} confidence
    </div>
    <div style="
        background:var(--secondary-background-color);
        border-radius:8px;
        height:10px;
        overflow:hidden;
        border:1px solid rgba(128,128,128,0.2);
    ">
        <div style="
            width:{conf_value*100:.1f}%;
            background:{color};
            height:100%;
            border-radius:8px;
        "></div>
    </div>
    <div style="text-align:right; color:var(--text-color); font-size:0.75rem; margin-top:4px;">
        {conf_value*100:.1f}%
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.metric("Fake probability", f"{fake_prob*100:.1f}%")
    col2.metric("Real probability", f"{real_prob*100:.1f}%")
    st.metric("Inference time", f"{inference_ms:.1f} ms")

    if heatmap is not None:
        high_activation = np.sum(heatmap > 0.7) / heatmap.size * 100
        st.markdown(f"""
        <div style="
            background:rgba(124,58,237,0.10);
            border:1px solid #7c3aed;
            border-radius:8px;
            padding:8px 14px;
            margin-top:12px;
            font-size:0.82rem;
            color:#7c3aed;
        ">
            🔥 High-activation region: <b>{high_activation:.1f}%</b> of image area
        </div>
        """, unsafe_allow_html=True)


# ── Load model — show_spinner=False removes the default "Running load_model()" ─
GDRIVE_FILE_ID = "1IlYSYufMdWM2bhBIVhYHxivhRd8eVW2X"
MODEL_PATH     = "deepfake_detector.keras"

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            import gdown
            gdown.download(
                f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
                MODEL_PATH,
                quiet=True
            )
        model = tf.keras.models.load_model(MODEL_PATH)
        dummy = tf.zeros((1, 224, 224, 3))
        model(dummy, training=False)
        last_conv = get_last_conv_layer(model)
        return model, last_conv
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None, None


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict_image(img_rgb, model, last_conv_name, threshold, show_gradcam, alpha):
    processed = preprocess_for_model(img_rgb)
    t0   = time.perf_counter()
    preds = model.predict(processed, verbose=0)
    inference_ms = (time.perf_counter() - t0) * 1000

    if preds.shape[-1] == 1:
        real_prob = float(preds[0][0])
        fake_prob = 1.0 - real_prob
    else:
        fake_prob = float(preds[0][1])
        real_prob = 1.0 - fake_prob

    is_fake = fake_prob >= threshold
    result  = {
        "is_fake":      is_fake,
        "fake_prob":    fake_prob,
        "real_prob":    real_prob,
        "inference_ms": inference_ms,
        "overlay":      None,
        "heatmap":      None
    }
    if show_gradcam:
        heatmap           = compute_gradcam(processed, model, last_conv_name)
        result["overlay"] = overlay_heatmap(img_rgb, heatmap, alpha=alpha)
        result["heatmap"] = heatmap
    return result


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Load model silently — no "Running load_model()" toast
    with st.spinner("Loading DeepShield…"):
        model, last_conv = load_model()
    if model is None:
        st.stop()

    if "history" not in st.session_state:
        st.session_state.history = []

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style="
            text-align:center;
            padding:16px 0 12px 0;
            border-bottom:1px solid rgba(128,128,128,0.2);
            margin-bottom:16px;
        ">
            <div style="font-size:1.8rem;">🛡️</div>
            <div style="font-size:1.05rem; font-weight:700; color:var(--text-color);">
                DeepShield
            </div>
            <div style="font-size:0.72rem; color:var(--text-color); opacity:0.5;">
                Media Authentication System
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**⚙️ Detection Settings**")
        threshold = st.slider(
            "Detection threshold", 0.1, 0.9, 0.5, 0.05,
            help="Higher = fewer false positives. Lower = more sensitive."
        )
        alpha        = st.slider("Grad-CAM opacity", 0.1, 0.9, 0.45, 0.05)
        show_gradcam = st.checkbox("Show Grad-CAM heatmap", value=True)

        st.markdown("---")
        st.markdown("**🧠 Model Information**")
        st.markdown(f"""
        <div style="
            background:var(--background-color);
            border:1px solid rgba(128,128,128,0.2);
            border-radius:10px;
            padding:12px 14px;
            font-size:0.78rem;
            color:var(--text-color);
            line-height:2;
        ">
            <span style="color:{BLUE}; font-weight:600;">Architecture</span>
            <br>EfficientNetB0<br>
            <span style="color:{BLUE}; font-weight:600;">Parameters</span>
            <br>{model.count_params():,}<br>
            <span style="color:{BLUE}; font-weight:600;">Input shape</span>
            <br>224 × 224 × 3<br>
            <span style="color:{BLUE}; font-weight:600;">Output</span>
            <br>Sigmoid (Binary)<br>
            <span style="color:{BLUE}; font-weight:600;">Last conv layer</span>
            <br>{last_conv}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"""
        <div style="font-size:0.72rem; color:var(--text-color); opacity:0.5; line-height:1.9;">
            CDS6334 Visual Information Processing<br>
            Trimester 2610 · Group Project<br>
            Test Accuracy:
            <span style="color:#22c55e; font-weight:600;">80.90%</span> ·
            AUC: <span style="color:#22c55e; font-weight:600;">0.9086</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Hero header ────────────────────────────────────────────────────────────
    render_header()

    tab_img, tab_vid, tab_batch, tab_history = st.tabs([
        "📷  Single Image",
        "🎥  Video Analysis",
        "📦  Batch Images",
        "📋  Session History"
    ])

    # ── TAB 1: Single image ───────────────────────────────────────────────────
    with tab_img:
        st.caption("Upload a face image to detect whether it is authentic or AI-generated.")
        uploaded = st.file_uploader(
            "Drop an image here",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )
        if uploaded is not None:
            img_pil = Image.open(uploaded).convert("RGB")
            img_rgb = np.array(img_pil)

            with st.spinner("Analysing image…"):
                r = predict_image(img_rgb, model, last_conv,
                                  threshold, show_gradcam, alpha)

            st.session_state.history.append({
                "Filename":         uploaded.name,
                "Verdict":          "⚠️ Deepfake" if r["is_fake"] else "✅ Authentic",
                "Fake probability": f"{r['fake_prob']*100:.1f}%",
                "Inference (ms)":   f"{r['inference_ms']:.1f}"
            })

            if show_gradcam and r["overlay"] is not None:
                col1, col2, col3 = st.columns([1.2, 1.2, 1])
                with col1:
                    st.caption("Original Image")
                    st.image(img_rgb, use_container_width=True)
                with col2:
                    st.caption("Grad-CAM Explanation")
                    st.image(r["overlay"], use_container_width=True)
                with col3:
                    render_verdict(r["is_fake"], r["fake_prob"],
                                   r["real_prob"], r["inference_ms"],
                                   r["heatmap"])
                    buf = io.BytesIO()
                    Image.fromarray(r["overlay"]).save(buf, format="PNG")
                    st.download_button(
                        "⬇️ Download Grad-CAM",
                        buf.getvalue(),
                        file_name="gradcam_result.png",
                        mime="image/png"
                    )
            else:
                col1, col2 = st.columns([1.5, 1])
                with col1:
                    st.caption("Original Image")
                    st.image(img_rgb, use_container_width=True)
                with col2:
                    render_verdict(r["is_fake"], r["fake_prob"],
                                   r["real_prob"], r["inference_ms"])

    # ── TAB 2: Video ──────────────────────────────────────────────────────────
    with tab_vid:
        st.caption("Upload a video to analyse frames for deepfake manipulation.")
        video_file = st.file_uploader(
            "Drop a video here",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed"
        )
        if video_file is not None:
            max_frames = st.slider("Max frames to analyse", 10, 100, 30, 5)
            if st.button("🔍 Analyse Video"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(video_file.read())
                    tmp_path = tmp.name

                cap          = cv2.VideoCapture(tmp_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps          = max(cap.get(cv2.CAP_PROP_FPS), 1)
                step         = max(1, total_frames // max_frames)
                frame_results, fake_probs = [], []
                progress_bar = st.progress(0, text="Processing frames…")
                sampled = idx = 0

                while cap.isOpened() and sampled < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if idx % step == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        r = predict_image(frame_rgb, model, last_conv,
                                          threshold, False, alpha)
                        fake_probs.append(r["fake_prob"])
                        frame_results.append({
                            "Frame":     idx,
                            "Time (s)":  round(idx / fps, 2),
                            "Fake prob": round(r["fake_prob"] * 100, 1),
                            "Verdict":   "⚠️ Deepfake" if r["is_fake"] else "✅ Authentic"
                        })
                        sampled += 1
                        progress_bar.progress(sampled / max_frames,
                                              text=f"Frame {idx} / {total_frames}")
                    idx += 1

                cap.release()
                os.unlink(tmp_path)
                progress_bar.empty()

                avg_fake     = float(np.mean(fake_probs))
                overall_fake = avg_fake >= threshold
                color_v      = "#ef4444" if overall_fake else "#22c55e"
                bg_v         = "rgba(239,68,68,0.10)" if overall_fake else "rgba(34,197,94,0.10)"
                label_v      = "⚠️ DEEPFAKE DETECTED" if overall_fake else "✅ AUTHENTIC"

                st.markdown(f"""
                <div style="background:{bg_v};border:2px solid {color_v};
                border-radius:12px;padding:16px 24px;text-align:center;margin:16px 0;">
                    <span style="font-size:1.3rem;font-weight:800;color:{color_v};">
                        {label_v}</span><br>
                    <span style="color:var(--text-color);font-size:0.85rem;">
                        Average fake probability: {avg_fake*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)

                df = pd.DataFrame(frame_results)
                st.line_chart(df.set_index("Time (s)")["Fake prob"],
                              y_label="Fake probability (%)")
                st.dataframe(df, use_container_width=True)
                st.info(f"Analysed {sampled} frames  |  "
                        f"Video duration: {total_frames/fps:.1f}s  |  "
                        f"Sampling rate: {sampled/(total_frames/fps):.1f} fps")

    # ── TAB 3: Batch ──────────────────────────────────────────────────────────
    with tab_batch:
        st.caption("Upload multiple images for bulk analysis. Results export as CSV.")
        files = st.file_uploader(
            "Drop images here",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if files:
            if st.button(f"🔍 Analyse all {len(files)} images"):
                results, total_time = [], 0.0
                prog = st.progress(0)

                for i, f in enumerate(files):
                    img_rgb = np.array(Image.open(f).convert("RGB"))
                    r = predict_image(img_rgb, model, last_conv,
                                      threshold, False, alpha)
                    total_time += r["inference_ms"]
                    results.append({
                        "Filename":         f.name,
                        "Verdict":          "⚠️ Deepfake" if r["is_fake"] else "✅ Authentic",
                        "Fake probability": f"{r['fake_prob']*100:.1f}%",
                        "Inference (ms)":   f"{r['inference_ms']:.0f}"
                    })
                    prog.progress((i + 1) / len(files))

                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                fake_count = sum(1 for r in results if "Deepfake" in r["Verdict"])
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total images",       len(results))
                c2.metric("Deepfakes detected", fake_count)
                c3.metric("Authentic",          len(results) - fake_count)
                c4.metric("Avg inference",      f"{total_time/len(results):.0f} ms")

                csv = df.to_csv(index=False).encode()
                st.download_button("⬇️ Download results CSV",
                                   csv, "batch_results.csv", "text/csv")

    # ── TAB 4: Session History ─────────────────────────────────────────────────
    with tab_history:
        st.caption("All images analysed in this session are logged here.")
        if not st.session_state.history:
            st.markdown("""
            <div style="
                text-align:center; padding:48px;
                background:var(--secondary-background-color);
                border-radius:12px;
                border:1px dashed rgba(128,128,128,0.3);
            ">
                <div style="font-size:2rem; margin-bottom:8px;">📋</div>
                <div style="color:var(--text-color);">No images analysed yet.</div>
                <div style="font-size:0.8rem; margin-top:4px; color:var(--text-color); opacity:0.5;">
                    Upload images in the Single Image or Batch tab to begin.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            df_history = pd.DataFrame(st.session_state.history)
            st.dataframe(df_history, use_container_width=True)

            fake_total = sum(1 for h in st.session_state.history
                             if "Deepfake" in h["Verdict"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Analysed", len(st.session_state.history))
            c2.metric("Deepfakes Found", fake_total)
            c3.metric("Authentic",       len(st.session_state.history) - fake_total)

            col_a, col_b = st.columns(2)
            with col_a:
                csv = df_history.to_csv(index=False).encode()
                st.download_button("⬇️ Download Session Report (CSV)",
                                   csv, "session_history.csv", "text/csv")
            with col_b:
                if st.button("🗑️ Clear History"):
                    st.session_state.history = []
                    st.rerun()


if __name__ == "__main__":
    main()
