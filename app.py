
import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Dermoscopic Image Quality Assistant",
    page_icon="🔬",
    layout="wide"
)


def pil_to_bgr(image):
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def assess_image(bgr):
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    under = float(np.mean(gray < 30) * 100)
    over = float(np.mean(gray > 245) * 100)

    # Rough dark-line indicator. Not a validated hair detector.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (17, 17)
    )
    blackhat = cv2.morphologyEx(
        gray, cv2.MORPH_BLACKHAT, kernel
    )
    artifact_ratio = float(
        np.mean(blackhat > 18) * 100
    )

    checks = {
        "Resolution": "Adequate" if min(h, w) >= 256 else "Low",
        "Brightness": "Within heuristic range" if 45 <= brightness <= 210 else "Review",
        "Contrast": "Within heuristic range" if contrast >= 25 else "Low",
        "Sharpness": "Within heuristic range" if sharpness >= 50 else "Potential blur",
        "Underexposure": "Low" if under < 15 else "Elevated",
        "Overexposure": "Low" if over < 10 else "Elevated",
        "Dark-line artefact indicator": "Low" if artifact_ratio < 3 else "Elevated"
    }

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
        "Width (px)": w,
        "Height (px)": h,
        "Mean brightness (0-255)": round(brightness, 2),
        "Grayscale contrast (std)": round(contrast, 2),
        "Sharpness (Laplacian variance)": round(sharpness, 2),
        "Underexposed pixels (%)": round(under, 2),
        "Overexposed pixels (%)": round(over, 2),
        "Dark-line artefact indicator (%)": round(artifact_ratio, 2),
        "Heuristic quality score (/100)": score,
        **checks
    }


def conservative_hair_mask(gray):
    """
    Experimental directional dark-line detection.
    Only elongated connected components are retained.
    This can still detect lesion structures, so use cautiously.
    """

    kernels = []

    # Horizontal and vertical line kernels
    horizontal = np.zeros((17, 17), dtype=np.uint8)
    cv2.line(horizontal, (0, 8), (16, 8), 1, 1)
    kernels.append(horizontal)

    vertical = np.zeros((17, 17), dtype=np.uint8)
    cv2.line(vertical, (8, 0), (8, 16), 1, 1)
    kernels.append(vertical)

    # Diagonal line kernels
    diagonal1 = np.zeros((17, 17), dtype=np.uint8)
    cv2.line(diagonal1, (0, 0), (16, 16), 1, 1)
    kernels.append(diagonal1)

    diagonal2 = np.zeros((17, 17), dtype=np.uint8)
    cv2.line(diagonal2, (0, 16), (16, 0), 1, 1)
    kernels.append(diagonal2)

    response = np.zeros_like(gray)

    for kernel in kernels:
        blackhat = cv2.morphologyEx(
            gray, cv2.MORPH_BLACKHAT, kernel
        )
        response = np.maximum(response, blackhat)

    _, binary = cv2.threshold(
        response, 25, 255, cv2.THRESH_BINARY
    )

    # Retain only elongated connected components.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    mask = np.zeros_like(gray)

    for i in range(1, count):
        x, y, w, h, area = stats[i]

        longest = max(w, h)
        shortest = max(1, min(w, h))
        aspect_ratio = longest / shortest

        if longest >= 10 and aspect_ratio >= 2.5 and area <= 1500:
            mask[labels == i] = 255

    # Minimal expansion to cover line edges.
    mask = cv2.dilate(
        mask,
        np.ones((2, 2), dtype=np.uint8),
        iterations=1
    )

    return mask


def preprocess(
    bgr,
    remove_dark_lines=False,
    apply_clahe=True,
    max_side=1024
):
    out = bgr.copy()

    h, w = out.shape[:2]
    scale = min(1.0, max_side / max(h, w))

    if scale < 1:
        out = cv2.resize(
            out,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    if remove_dark_lines:
        gray = cv2.cvtColor(
            out, cv2.COLOR_BGR2GRAY
        )

        mask = conservative_hair_mask(gray)

        # Avoid inpainting if the detected mask is too extensive.
        mask_ratio = np.mean(mask > 0)

        if 0 < mask_ratio < 0.015:
            out = cv2.inpaint(
                out, mask, 2, cv2.INPAINT_TELEA
            )

    if apply_clahe:
        lab = cv2.cvtColor(
            out, cv2.COLOR_BGR2LAB
        )

        l, a, b = cv2.split(lab)

        # Reduced enhancement strength
        clahe = cv2.createCLAHE(
            clipLimit=1.2,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        out = cv2.cvtColor(
            cv2.merge((l, a, b)),
            cv2.COLOR_LAB2BGR
        )

    return out


def bgr_to_pil(bgr):
    return Image.fromarray(
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    )


st.title(
    "Dermoscopic Image Quality Assessment and Preprocessing Assistant"
)

st.caption(
    "A research prototype for image-quality screening and "
    "optional preprocessing — not a diagnostic system."
)


with st.sidebar:
    st.header("Preprocessing options")

    remove_lines = st.checkbox(
        "Experimental dark-line / hair reduction",
        value=False
    )

    clahe_on = st.checkbox(
        "Apply mild CLAHE contrast enhancement",
        value=True
    )

    max_side = st.selectbox(
        "Maximum output dimension",
        [512, 768, 1024, 1536],
        index=2
    )

    if remove_lines:
        st.warning(
            "Experimental hair removal may alter lesion structures. "
            "Always compare the output with the original."
        )

    st.divider()

    st.caption(
        "Quality thresholds and the score are heuristic "
        "and have not been clinically validated."
    )


uploaded = st.file_uploader(
    "Upload a dermoscopic image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)


if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")

        original = pil_to_bgr(image)

        metrics = assess_image(original)

        processed = preprocess(
            original,
            remove_lines,
            clahe_on,
            max_side
        )

        left, right = st.columns(2)

        with left:
            st.subheader("Original image")
            st.image(image, use_container_width=True)

        with right:
            st.subheader("Processed preview")
            st.image(
                bgr_to_pil(processed),
                use_container_width=True
            )

        st.subheader("Image quality assessment")

        score = metrics[
            "Heuristic quality score (/100)"
        ]

        st.metric(
            "Heuristic quality score",
            f"{score}/100"
        )

        st.warning(
            "This score is an engineering heuristic, "
            "not a validated measure of diagnostic suitability. "
            "Review the image visually."
        )

        display = pd.DataFrame([
            {"Parameter": k, "Assessment": v}
            for k, v in metrics.items()
        ])

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        csv = pd.DataFrame([metrics]).to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download assessment (CSV)",
            csv,
            "image_quality_assessment.csv",
            "text/csv"
        )

        buf = io.BytesIO()

        bgr_to_pil(processed).save(
            buf, format="PNG"
        )

        st.download_button(
            "Download processed image (PNG)",
            buf.getvalue(),
            "processed_dermoscopic_image.png",
            "image/png"
        )

    except Exception as exc:
        st.error(
            f"Could not process this file: {exc}"
        )

else:
    st.info(
        "Upload one image to view its quality indicators "
        "and processed preview."
    )


st.divider()

st.markdown(
    "**Limitations:** Hair detection and preprocessing are "
    "experimental and may alter lesion structures. CLAHE may "
    "change image texture and contrast. Preserve the original "
    "image, inspect all outputs, and do not use this prototype "
    "to make clinical decisions."
)
