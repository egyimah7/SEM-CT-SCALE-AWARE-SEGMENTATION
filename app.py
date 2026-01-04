# app.py
import io
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from analysis import mask_to_png_black_crack, prob_to_png
from predict import load_model_any, predict_single

import os
import urllib.request

FILE_ID = "1vMtdPhel-Bq1YhwArYqdPHoFJLEPmiqx"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
LOCAL_MODEL_PATH = "models/best_model.keras"

os.makedirs("models", exist_ok=True)
if not os.path.exists(LOCAL_MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, LOCAL_MODEL_PATH)


st.set_page_config(page_title="SEM/CT Scale Aware Segmentation", layout="wide")
st.title("SEM/CT Scale Aware Segmentation")
st.caption(
    "Upload a SEM/CT slice + enter its resolution (µm/px). "
    "Optional: upload a GT mask (black=crack/pore) to compute IoU/F1/Precision/Recall."
)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Model")
model_path = st.sidebar.text_input("Model path", "models/best_model.keras")


st.sidebar.header("Resolution")
res_um_px = st.sidebar.number_input("Image resolution (µm/px)", min_value=1e-6, value=0.084, format="%.6f")
canonical_res = st.sidebar.number_input("Canonical resolution (µm/px)", min_value=1e-6, value=2.65, format="%.6f")

st.sidebar.header("Thresholding (hysteresis)")
t_low = st.sidebar.slider("Low threshold (t_low)", 0.0, 1.0, 0.35, 0.01)
t_high = st.sidebar.slider("High threshold (t_high)", 0.0, 1.0, 0.70, 0.01)

st.sidebar.header("Post-processing")
min_obj_px = st.sidebar.number_input("Remove tiny objects (< px)", min_value=0, value=10, step=1)
close_radius = st.sidebar.number_input("Closing radius (px)", min_value=0, value=1, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: If your prediction misses thin cracks, try lowering t_low/t_high slightly and reducing min_obj_px.")

# -----------------------------
# Upload inputs
# -----------------------------
colA, colB = st.columns(2)
with colA:
    up_img = st.file_uploader("Upload SEM/CT slice (PNG/JPG/TIF)", type=["png", "jpg", "jpeg", "tif", "tiff"])
with colB:
    up_gt = st.file_uploader("Optional: Upload GT mask (black=crack/pore)", type=["png", "jpg", "jpeg", "tif", "tiff"])

# -----------------------------
# Load model once (cached)
# -----------------------------
@st.cache_resource(show_spinner=False)
def _load_cached_model(path: str):
    return load_model_any(path)

try:
    model_runner = _load_cached_model(model_path)
except Exception as e:
    st.error(f"Failed to load model from: {model_path}\n\n{e}")
    st.stop()

if up_img is None:
    st.info("Upload an image to start.")
    st.stop()

img_bytes = up_img.read()
gt_bytes = up_gt.read() if up_gt is not None else None

# -----------------------------
# Run prediction
# -----------------------------
with st.spinner("Running prediction..."):
    out = predict_single(
        model_runner=model_runner,
        image_bytes=img_bytes,
        res_um_px=float(res_um_px),
        canonical_res=float(canonical_res),
        t_low=float(t_low),
        t_high=float(t_high),
        min_obj_px=int(min_obj_px),
        close_radius=int(close_radius),
        gt_mask_bytes=gt_bytes,
    )

img01 = out["img01"]           # float [0,1] HxW
prob = out["prob"]             # float [0,1] HxW
pred01 = out["pred"]           # uint8 0/1 HxW  (1 = crack/pore)
metrics = out["metrics"]       # dict or None
morph = out["morph_summary"]   # dict (unit-aware)
df_obj = out["df_objects"]     # DataFrame (unit-aware columns)

# -----------------------------
# Prepare visuals
# -----------------------------
raw = np.asarray(img01)

# squeeze (H,W,1) -> (H,W)
if raw.ndim == 3 and raw.shape[-1] == 1:
    raw = raw[..., 0]

# convert to uint8 for display
if raw.dtype != np.uint8:
    rmax = float(raw.max())
    if rmax <= 1.5:  # likely [0,1]
        raw_u8 = (np.clip(raw, 0, 1) * 255).astype(np.uint8)
    else:            # likely [0,255] float
        raw_u8 = np.clip(raw, 0, 255).astype(np.uint8)
else:
    raw_u8 = raw

prob_u8 = prob_to_png(prob)
pred_u8 = mask_to_png_black_crack(pred01)  # crack/pore black; matrix white

overlay_rgb = np.stack([raw_u8, raw_u8, raw_u8], axis=-1).astype(np.uint8)
overlay_rgb[pred01 > 0] = [255, 200, 0]

# -----------------------------
# Display images
# -----------------------------
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Input")
    st.image(raw_u8, clamp=True)
with c2:
    st.subheader("Prob (model output)")
    st.image(prob_u8, clamp=True)
with c3:
    st.subheader("Pred mask (black=crack/pore)")
    st.image(pred_u8, clamp=True)

st.subheader("Overlay (prediction on input)")
st.image(overlay_rgb, clamp=True)

# -----------------------------
# Metrics & Morphology
# -----------------------------
st.subheader("Quantitative outputs")
mcol1, mcol2 = st.columns(2)

with mcol1:
    st.markdown("### Morphology (from prediction)")
    st.caption("All lengths in µm, areas in µm² (see *_um and *_um2 fields).")
    st.json(morph)

with mcol2:
    st.markdown("### Validation metrics")
    if metrics is None:
        st.info("No GT mask uploaded → metrics not computed.")
        st.caption("For publishable metrics (IoU/F1/etc.), upload a GT mask for the same slice.")
    else:
        st.json(metrics)

if df_obj is not None and len(df_obj) > 0:
    st.markdown("### Object table (prediction)")
    st.caption("Unit-aware columns included: area_um2, perimeter_um, major_axis_length_um, minor_axis_length_um, eq_diam_um, aperture_um_proxy.")
    st.dataframe(df_obj, use_container_width=True)
else:
    st.caption("No objects detected (or removed by settings).")

# -----------------------------
# Download ZIP report
# -----------------------------
def _png_bytes(arr: np.ndarray) -> bytes:
    im = Image.fromarray(arr)
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()

def _dict_to_csv_bytes(d: dict) -> bytes:
    return pd.DataFrame([d]).to_csv(index=False).encode("utf-8")

def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

zip_buf = io.BytesIO()
with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
    z.writestr("images/raw.png", _png_bytes(raw_u8))
    z.writestr("images/prob.png", _png_bytes(prob_u8))
    z.writestr("images/pred_black_crack.png", _png_bytes(pred_u8))
    z.writestr("images/overlay.png", _png_bytes(overlay_rgb))

    z.writestr("tables/morphology_summary.csv", _dict_to_csv_bytes(morph))
    if metrics is not None:
        z.writestr("tables/metrics.csv", _dict_to_csv_bytes(metrics))
    if df_obj is not None and len(df_obj) > 0:
        z.writestr("tables/objects.csv", _df_to_csv_bytes(df_obj))

zip_buf.seek(0)

st.download_button(
    label="Download report bundle (ZIP)",
    data=zip_buf,
    file_name="sem_ct_scale_aware_report.zip",
    mime="application/zip",
)




