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

    penalties = 0
    penalties += 15 if min(h, w) < 256 else 0
    penalties += 15 if not (45 <= brightness <= 210) else 0
    penalties += 15 if contrast < 25 else 0
    penalties += 20 if sharpness < 50 else 0
    penalties += 10 if under >= 15 else 0
    penalties += 10 if over >= 10 else 0

    score = max(0, 100 - penalties)

    return {
        "Width (px)": w,
        "Height (px)": h,
        "Mean brightness (0-255)": round(brightness, 2),
        "Grayscale contrast (std)": round(contrast, 2),
        "Sharpness (Laplacian variance)": round(sharpness, 2),
        "Underexposed pixels (%)": round(under, 2),
        "Overexposed pixels (%)": round(over, 2),
        "Heuristic quality score (/100)": score,
        **checks,
    }


def make_line_kernel(length, angle_degrees):
    """Create a thin directional structuring element for hair-like lines."""
    kernel = np.zeros((length, length), dtype=np.uint8)
    centre = length // 2
    radians = np.deg2rad(angle_degrees)
    dx = int(round((length // 2 - 1) * np.cos(radians)))
    dy = int(round((length // 2 - 1) * np.sin(radians)))
    cv2.line(
        kernel,
        (centre - dx, centre - dy),
        (centre + dx, centre + dy),
        1,
        thickness=1,
    )
    return kernel


def detect_conservative_hair_mask(bgr):
    """
    Conservative multi-direction black-hat detector.

    It retains only small, thin, elongated connected components. If the
    resulting mask is too large, it returns an empty mask rather than
    inpainting broad image/lesion regions.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    image_area = h * w

    responses = []
    for length in (17, 25, 33):
        for angle in (0, 45, 90, 135):
            kernel = make_line_kernel(length, angle)
            response = cv2.morphologyEx(
                gray, cv2.MORPH_BLACKHAT, kernel
            )
            responses.append(response)

    response = np.maximum.reduce(responses)

    # Otsu adapts to the response range; the floor prevents weak texture
    # responses from becoming a very large mask.
    otsu_value, _ = cv2.threshold(
        response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold_value = max(10, int(otsu_value))
    binary = (response >= threshold_value).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    candidates = []
    for label in range(1, count):
        x, y, bw, bh, area = stats[label]

        if area < 5:
            continue
        if area > max(1200, int(image_area * 0.0015)):
            continue

        long_side = max(bw, bh)
        short_side = max(1, min(bw, bh))

        # Hair is generally thin and elongated. PCA also handles diagonal
        # components whose axis-aligned bounding box is less informative.
        ys, xs = np.where(labels == label)
        if len(xs) < 5:
            continue

        points = np.column_stack((xs, ys)).astype(np.float32)
        covariance = np.cov(points, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        elongation = float(
            eigenvalues[-1] / max(eigenvalues[0], 1e-3)
        )

        if long_side < 10:
            continue
        if short_side > 18:
            continue
        if elongation < 2.0:
            continue

        mean_strength = float(np.mean(response[labels == label]))
        candidates.append((mean_strength, label, area))

    mask = np.zeros((h, w), dtype=np.uint8)

    # Limit the selected components before dilation. This prevents the
    # inpainting mask from swallowing large portions of the lesion.
    max_component_pixels = int(image_area * 0.012)
    used_pixels = 0

    candidates.sort(key=lambda item: item[0], reverse=True)
    for _, label, area in candidates:
        if used_pixels + area > max_component_pixels:
            continue
        mask[labels == label] = 255
        used_pixels += area

    if used_pixels == 0:
        return mask, 0.0, "No confident thin hair-like structures detected."

    # A single small dilation covers the hair edges without expanding the
    # mask aggressively.
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    coverage = float(np.mean(mask > 0) * 100)

    # Safety fallback: do not inpaint if detection covers too much of image.
    if coverage > 2.5:
        return (
            np.zeros((h, w), dtype=np.uint8),
            coverage,
            "Hair removal skipped: detected region was too large. Original preserved.",
        )

    return mask, coverage, "Conservative hair-like mask applied."


def preprocess(
    bgr,
    remove_dark_lines=True,
    apply_clahe=True,
    max_side=1024,
):
    out = bgr.copy()
    h, w = out.shape[:2]

    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        out = cv2.resize(
            out,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    hair_status = "Hair reduction disabled."
    mask_coverage = 0.0

    if remove_dark_lines:
        mask, mask_coverage, hair_status = detect_conservative_hair_mask(out)

        if np.any(mask):
            # Small inpainting radius reduces broad texture replacement.
            out = cv2.inpaint(out, mask, 2, cv2.INPAINT_TELEA)

    if apply_clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Mild CLAHE: less aggressive than the earlier clipLimit=2.0 setting.
        clahe = cv2.createCLAHE(
            clipLimit=1.5,
            tileGridSize=(8, 8),
        )
        l_channel = clahe.apply(l_channel)

        out = cv2.cvtColor(
            cv2.merge((l_channel, a_channel, b_channel)),
            cv2.COLOR_LAB2BGR,
        )

    return out, hair_status, mask_coverage


st.title("Dermoscopic Image Quality Assessment and Preprocessing Assistant")
st.caption(
    "A research prototype for image-quality screening and optional "
    "preprocessing — not a diagnostic system."
)

with st.sidebar:
    st.header("Preprocessing options")
    remove_lines = st.checkbox(
        "Experimental hair reduction",
        value=True,
        help=(
            "Uses a conservative multi-direction detector. Some hairs may "
            "remain; uncertain or overly large masks are rejected."
        ),
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

    st.divider()
    st.caption(
        "Quality thresholds and the score are heuristic and have not been "
        "clinically validated."
    )
    st.warning(
        "Preprocessing can alter lesion structures. Always compare the "
        "processed image with the original."
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
        processed, hair_status, mask_coverage = preprocess(
            original,
            remove_dark_lines=remove_lines,
            apply_clahe=clahe_on,
            max_side=max_side,
        )

        left, right = st.columns(2)
        with left:
            st.subheader("Original image")
            st.image(image, use_container_width=True)

        with right:
            st.subheader("Processed preview")
            st.image(bgr_to_pil(processed), use_container_width=True)

        if remove_lines:
            st.info(
                f"{hair_status} "
                f"Mask coverage before inpainting: {mask_coverage:.2f}%."
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
    "or identify dark lesion structures. Inpainting and CLAHE can change "
    "image appearance. Preserve the original image, inspect all outputs, "
    "and do not use this prototype to make clinical decisions."
)
