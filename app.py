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


def bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def assess_image(bgr):
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    under = float(np.mean(gray < 30) * 100)
    over = float(np.mean(gray > 245) * 100)

    checks = {
        "Resolution": "Adequate" if min(h, w) >= 256 else "Low",
        "Brightness": "Within heuristic range" if 45 <= brightness <= 210 else "Review",
        "Contrast": "Within heuristic range" if contrast >= 25 else "Low",
        "Sharpness": "Within heuristic range" if sharpness >= 50 else "Potential blur",
        "Underexposure": "Low" if under < 15 else "Elevated",
        "Overexposure": "Low" if over < 10 else "Elevated",
    }

    penalties = (
        (15 if min(h, w) < 256 else 0)
        + (15 if not (45 <= brightness <= 210) else 0)
        + (15 if contrast < 25 else 0)
        + (20 if sharpness < 50 else 0)
        + (10 if under >= 15 else 0)
        + (10 if over >= 10 else 0)
    )

    return {
        "Width (px)": w,
        "Height (px)": h,
        "Mean brightness (0-255)": round(brightness, 2),
        "Grayscale contrast (std)": round(contrast, 2),
        "Sharpness (Laplacian variance)": round(sharpness, 2),
        "Underexposed pixels (%)": round(under, 2),
        "Overexposed pixels (%)": round(over, 2),
        "Heuristic quality score (/100)": max(0, 100 - penalties),
        **checks,
    }


def line_kernel(length, angle):
    """Create a one-pixel-wide line kernel at a specified orientation."""
    size = length
    kernel = np.zeros((size, size), dtype=np.uint8)
    centre = size // 2
    radius = max(1, centre - 1)
    radians = np.deg2rad(angle)
    dx = int(round(radius * np.cos(radians)))
    dy = int(round(radius * np.sin(radians)))
    cv2.line(
        kernel,
        (centre - dx, centre - dy),
        (centre + dx, centre + dy),
        1,
        thickness=1,
    )
    return kernel


def detect_hair_mask(bgr, sensitivity=10, max_mask_percent=4.0):
    """
    Multi-scale, multi-direction dark-line detector.

    Lower sensitivity values detect weaker/dimmer lines and may also select
    lesion structures. The connected-component checks reduce broad regions,
    but cannot perfectly distinguish hairs from all dermoscopic structures.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    image_area = h * w

    # Use several line lengths and orientations to capture fine and thicker
    # hairs running in different directions.
    response = np.zeros_like(gray)
    for length in (9, 13, 17, 23, 29):
        for angle in (0, 30, 60, 90, 120, 150):
            kernel = line_kernel(length, angle)
            blackhat = cv2.morphologyEx(
                gray, cv2.MORPH_BLACKHAT, kernel
            )
            response = np.maximum(response, blackhat)

    # A user-controlled absolute response threshold. Lower = more sensitive.
    _, binary = cv2.threshold(
        response, int(sensitivity), 255, cv2.THRESH_BINARY
    )

    # Join short gaps in hair traces without expanding the mask substantially.
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    components = []
    for label in range(1, count):
        x, y, bw, bh, area = stats[label]
        if area < 4:
            continue

        ys, xs = np.where(labels == label)
        if len(xs) < 4:
            continue

        points = np.column_stack((xs, ys)).astype(np.float32)
        covariance = np.cov(points, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        elongation = float(
            eigenvalues[-1] / max(eigenvalues[0], 1e-3)
        )

        long_side = max(bw, bh)
        short_side = max(1, min(bw, bh))

        # Keep narrow elongated traces. Slightly relaxed filters compared with
        # the previous version so faint/fragmented hairs are not discarded.
        if long_side < 7:
            continue
        if short_side > 14:
            continue
        if elongation < 1.6:
            continue
        if area > max(300, int(image_area * 0.001)):
            continue

        strength = float(np.mean(response[labels == label]))
        components.append((strength, label, area))

    # Prefer stronger line responses but permit more components than before.
    components.sort(key=lambda item: item[0], reverse=True)
    mask = np.zeros((h, w), dtype=np.uint8)
    pixel_budget = int(image_area * (max_mask_percent / 100.0))
    used = 0

    for _, label, area in components:
        if used + area > pixel_budget:
            continue
        mask[labels == label] = 255
        used += area

    # Expand slightly to cover hair edges. Keep this small to avoid lesion loss.
    if np.any(mask):
        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

    coverage = float(np.mean(mask > 0) * 100)
    return mask, coverage, response


def overlay_mask(bgr, mask):
    """Overlay detected pixels in red for transparent inspection."""
    overlay = bgr.copy()
    red = np.zeros_like(bgr)
    red[:, :] = (0, 0, 255)
    selected = mask > 0
    overlay[selected] = cv2.addWeighted(
        bgr[selected], 0.35, red[selected], 0.65, 0
    )
    return overlay


def preprocess(bgr, remove_hair=True, apply_clahe=True,
               max_side=1024, sensitivity=10):
    out = bgr.copy()
    h, w = out.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        out = cv2.resize(
            out,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    mask = np.zeros(out.shape[:2], dtype=np.uint8)
    coverage = 0.0
    response = np.zeros(out.shape[:2], dtype=np.uint8)

    if remove_hair:
        mask, coverage, response = detect_hair_mask(
            out, sensitivity=sensitivity
        )
        if np.any(mask):
            # Radius 2 limits the amount of surrounding texture replaced.
            out = cv2.inpaint(out, mask, 2, cv2.INPAINT_TELEA)

    if apply_clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=1.5,
            tileGridSize=(8, 8),
        )
        l_channel = clahe.apply(l_channel)
        out = cv2.cvtColor(
            cv2.merge((l_channel, a_channel, b_channel)),
            cv2.COLOR_LAB2BGR,
        )

    return out, mask, coverage, response


st.title("Dermoscopic Image Quality Assessment and Preprocessing Assistant")
st.caption(
    "A research prototype for image-quality screening and optional "
    "preprocessing — not a diagnostic system."
)

with st.sidebar:
    st.header("Preprocessing options")
    remove_hair = st.checkbox(
        "Experimental hair reduction",
        value=True,
    )
    clahe_on = st.checkbox(
        "Apply mild CLAHE contrast enhancement",
        value=True,
    )
    max_side = st.selectbox(
        "Maximum output dimension",
        [512, 768, 1024, 1536],
        index=2,
    )

    sensitivity = st.slider(
        "Hair detection sensitivity",
        min_value=4,
        max_value=24,
        value=10,
        step=1,
        help=(
            "Lower values detect more faint lines but may also select lesion "
            "structures. Increase the value if too much of the lesion is marked."
        ),
        disabled=not remove_hair,
    )

    st.divider()
    st.caption(
        "Quality thresholds and the score are heuristic and have not been "
        "clinically validated."
    )
    st.warning(
        "Hair removal can alter lesion structures. Always compare the "
        "processed output with the original."
    )

uploaded = st.file_uploader(
    "Upload a dermoscopic image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
)

if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
        original = pil_to_bgr(image)
        metrics = assess_image(original)

        processed, mask, coverage, response = preprocess(
            original,
            remove_hair=remove_hair,
            apply_clahe=clahe_on,
            max_side=max_side,
            sensitivity=sensitivity,
        )

        left, right = st.columns(2)
        with left:
            st.subheader("Original image")
            st.image(image, use_container_width=True)
        with right:
            st.subheader("Processed preview")
            st.image(bgr_to_pil(processed), use_container_width=True)

        if remove_hair:
            st.info(
                f"Detected mask coverage: {coverage:.2f}% of image area. "
                "This is mask coverage, not a percentage of hairs removed."
            )

            with st.expander("Inspect hair-detection mask", expanded=True):
                st.caption(
                    "Red pixels are the regions selected for inpainting. "
                    "If lesion areas are marked, increase sensitivity "
                    "(a higher value is more selective) or disable hair removal."
                )
                st.image(
                    bgr_to_pil(overlay_mask(
                        cv2.resize(
                            original,
                            (mask.shape[1], mask.shape[0]),
                            interpolation=cv2.INTER_AREA,
                        ) if original.shape[:2] != mask.shape else original,
                        mask,
                    )),
                    caption="Detected regions overlaid in red",
                    use_container_width=True,
                )
                st.image(
                    Image.fromarray(mask),
                    caption="Binary detection mask",
                    use_container_width=True,
                )

        st.subheader("Image quality assessment")
        score = metrics["Heuristic quality score (/100)"]
        st.metric("Heuristic quality score", f"{score}/100")
        st.warning(
            "This score is an engineering heuristic, not a validated "
            "measure of diagnostic suitability. Review the image visually."
        )

        display = pd.DataFrame(
            [
                {"Parameter": key, "Assessment": value}
                for key, value in metrics.items()
            ]
        )
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = (
            pd.DataFrame([metrics])
            .to_csv(index=False)
            .encode("utf-8")
        )
        st.download_button(
            "Download assessment (CSV)",
            csv_data,
            "image_quality_assessment.csv",
            "text/csv",
        )

        buffer = io.BytesIO()
        bgr_to_pil(processed).save(buffer, format="PNG")
        st.download_button(
            "Download processed image (PNG)",
            buffer.getvalue(),
            "processed_dermoscopic_image.png",
            "image/png",
        )

    except Exception as exc:
        st.error(f"Could not process this file: {exc}")
else:
    st.info(
        "Upload one image to view its quality indicators and processed preview."
    )

st.divider()
st.markdown(
    "**Limitations:** Hair detection is heuristic and may miss some hairs "
    "or select dark lesion structures. Inpainting and CLAHE can change "
    "image appearance. Preserve the original image, inspect the mask and "
    "all outputs, and do not use this prototype to make clinical decisions."
)
