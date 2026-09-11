# DDC Book Classifier — Run Guide

This document provides step-by-step instructions on how to set up, run, and test the DDC Book Classifier pipeline locally.

## Prerequisites

- **Python 3.9+** (Anaconda/Miniconda or standard Python)
- **Tesseract OCR**: You must have Tesseract installed on your system.
  - Windows: Download the installer from UB-Mannheim (e.g., `tesseract-ocr-w64-setup-5.3.0.xxxx.exe`) and install it. Add `C:\Program Files\Tesseract-OCR` to your System PATH.
  - macOS: `brew install tesseract`
  - Linux: `sudo apt-get install tesseract-ocr`
- **Git**

## 1. Setup the Environment

It's highly recommended to use a virtual environment to manage dependencies.

**Using Conda:**
```bash
conda create -n ddc-env python=3.10
conda activate ddc-env
```

**Using venv:**
```bash
python -m venv ddc-env
# Windows:
ddc-env\Scripts\activate
# macOS/Linux:
source ddc-env/bin/activate
```

## 2. Install Dependencies

Install the required Python packages from the repository root:

```bash
pip install -r requirements.txt
```

## 3. Configure the Environment Variables

The classification ensemble uses Google's Gemini Vision model alongside a local TF-IDF model. You must provide a Gemini API key.

1. Create a file named `.env` in the root of the repository.
2. Add your Gemini API key to it:

```env
GEMINI_API_KEY="your_api_key_here"
```

## 4. Run the Pipeline Server

The backend runs on FastAPI and serves the frontend web app directly.

Start the local server using `uvicorn`:

```bash
uvicorn src.api:app --reload
```

You will see output indicating that the server is running on `http://127.0.0.1:8000`.

## 5. Use the Web Interface

1. Open your web browser and navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000).
2. You will see the **Emerald Library** themed interface.
3. **Upload or Capture**: Provide a picture of a book's **Front Page/Cover** (required). You can also optionally provide the Summary Page and Index Page.
4. Click **Classify This Book**.
5. The pipeline will process the images using Tesseract, extract fields via Gemini, and route the text through the ensemble TF-IDF classifier.
6. The results, confidence probabilities, and the routing outcome (Auto-Accept, Needs Review, or Reject) will be displayed.

## 6. Training the TF-IDF Model (Optional)

If you have updated the underlying dataset or changed the pipeline configuration, you can retrain the baseline TF-IDF model locally.

Run the training script from the repository root:

```bash
python -m src.ocr_pipeline.train_baseline
```

This script will run a `GridSearchCV` over the dataset, calibrate the probabilities, and serialize the upgraded model to the `models/` directory for the API to use.

## 7. Running the Test Suite

To ensure the pipeline is functioning correctly and regressions haven't been introduced, run the `pytest` suite:

```bash
python -m pytest tests/ -v
```
