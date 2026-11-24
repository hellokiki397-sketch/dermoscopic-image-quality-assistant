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
