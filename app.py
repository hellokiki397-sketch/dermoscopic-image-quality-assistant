import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Dermoscopic Image Quality Assistant", page_icon="🔬", layout="wide")

def pil_to_bgr(image):
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def assess_image(bgr):
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    under = float(np.mean(gray < 30) * 100)
    over = float(np.mean(gray > 245) * 100)

    # A black-hat response highlights thin, dark structures; this is only a rough
    # hair/artifact indicator and may also respond to lesion structures.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    artifact_ratio = float(np.mean(blackhat > 18) * 100)

    checks = {
        "Resolution": "Adequate" if min(h, w) >= 256 else "Low",
        "Brightness": "Within heuristic range" if 45 <= brightness <= 210 else "Review",
        "Contrast": "Within heuristic range" if contrast >= 25 else "Low",
        "Sharpness": "Within heuristic range" if sharpness >= 50 else "Potential blur",
        "Underexposure": "Low" if under < 15 else "Elevated",
        "Overexposure": "Low" if over < 10 else "Elevated",
        "Dark-line artefact indicator": "Low" if artifact_ratio < 3 else "Elevated"
    }

    # Transparent heuristic score; not a clinically validated quality metric.
    penalties = 0
    penalties += 15 if min(h, w) < 256 else 0
    penalties += 15 if not (45 <= brightness <= 210) else 0
    penalties += 15 if contrast < 25 else 0
    penalties += 20 if sharpness < 50 else 0
    penalties += 10 if under >= 15 else 0
    penalties += 10 if over >= 10 else 0
    penalties += 15 if artifact_ratio >= 3 else 0
    score = max(0, 100 - penalties)

    return {
        "Width (px)": w, "Height (px)": h,
        "Mean brightness (0-255)": round(brightness, 2),
        "Grayscale contrast (std)": round(contrast, 2),
        "Sharpness (Laplacian variance)": round(sharpness, 2),
        "Underexposed pixels (%)": round(under, 2),
        "Overexposed pixels (%)": round(over, 2),
        "Dark-line artefact indicator (%)": round(artifact_ratio, 2),
        "Heuristic quality score (/100)": score,
        **checks
    }

def preprocess(bgr, remove_dark_lines=True, apply_clahe=True, max_side=1024):
    out = bgr.copy()
    h, w = out.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        out = cv2.resize(out, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    if remove_dark_lines:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, 12, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        out = cv2.inpaint(out, mask, 3, cv2.INPAINT_TELEA)

    if apply_clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return out

def bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

st.title("Dermoscopic Image Quality Assessment and Preprocessing Assistant")
st.caption("A research prototype for image-quality screening and optional preprocessing — not a diagnostic system.")

with st.sidebar:
    st.header("Preprocessing options")
    remove_lines = st.checkbox("Attempt dark-line / hair artefact reduction", value=True)
    clahe_on = st.checkbox("Apply CLAHE contrast enhancement", value=True)
    max_side = st.selectbox("Maximum output dimension", [512, 768, 1024, 1536], index=2)
    st.divider()
    st.caption("All quality thresholds and the score are heuristic. They have not been clinically validated.")

uploaded = st.file_uploader("Upload a dermoscopic image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
        original = pil_to_bgr(image)
        metrics = assess_image(original)
        processed = preprocess(original, remove_lines, clahe_on, max_side)

        left, right = st.columns(2)
        with left:
            st.subheader("Original image")
            st.image(image, use_container_width=True)
        with right:
            st.subheader("Processed preview")
            st.image(bgr_to_pil(processed), use_container_width=True)

        st.subheader("Image quality assessment")
        score = metrics["Heuristic quality score (/100)"]
        st.metric("Heuristic quality score", f"{score}/100")
        st.warning("This score is an engineering heuristic, not a validated measure of diagnostic suitability. Review the image visually.")
        display = pd.DataFrame(
            [{"Parameter": k, "Assessment": v} for k, v in metrics.items()]
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

        csv = pd.DataFrame([metrics]).to_csv(index=False).encode("utf-8")
        st.download_button("Download assessment (CSV)", csv, "image_quality_assessment.csv", "text/csv")

        buf = io.BytesIO()
        bgr_to_pil(processed).save(buf, format="PNG")
        st.download_button("Download processed image (PNG)", buf.getvalue(),
                           "processed_dermoscopic_image.png", "image/png")
    except Exception as exc:
        st.error(f"Could not process this file: {exc}")
else:
    st.info("Upload one image to view its quality indicators and processed preview.")

st.divider()
st.markdown("**Limitations:** The dark-line detector may also respond to lesion structures. CLAHE and inpainting can alter image appearance. Preserve the original image, inspect all outputs, and do not use this prototype to make clinical decisions.")
