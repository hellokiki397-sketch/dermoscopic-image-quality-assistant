# Dermoscopic Image Quality Assessment and Preprocessing Assistant

A lightweight Streamlit prototype that:
- reads a dermoscopic image,
- computes image-quality indicators (resolution, brightness, contrast, sharpness, exposure),
- estimates dark-line artefact prevalence using a morphological black-hat response,
- creates an optional CLAHE-enhanced and inpainted preview,
- exports a CSV assessment and processed PNG.

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Open the local URL printed by Streamlit (normally http://localhost:8501).

## Deploy
Push this folder to a GitHub repository and deploy `app.py` using Streamlit Community Cloud. Set the repository root to this project folder. Install dependencies from `requirements.txt`.

## Important limitations
- This is an engineering prototype, not a diagnostic model.
- Thresholds and the 0–100 score are heuristic and have not been validated against expert image-quality labels.
- Morphological hair/artifact detection may identify legitimate lesion structures.
- Inpainting and contrast enhancement can change image appearance. Keep the original image and inspect every processed output.
- No patient data is uploaded to a server by this code except through the hosting environment chosen by the operator. Do not upload identifiable or sensitive images to a public deployment without appropriate approvals.
