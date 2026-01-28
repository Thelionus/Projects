# Image & Document Fraud Detector

Detects manipulations in images and PDFs — Photoshop edits, altered text/names/amounts, copy-paste forgery, double compression, and more.

## Detection Techniques (10 analyses)

| # | Technique | What It Detects |
|---|-----------|-----------------|
| 1 | **Error Level Analysis (ELA)** | Regions re-saved at different JPEG quality (edited areas) |
| 2 | **Multi-Quality ELA** | Confirms ELA findings across multiple quality levels |
| 3 | **Noise Consistency** | Blocks with different noise patterns (pasted from another source) |
| 4 | **Copy-Move Detection** | Duplicated/cloned regions within the same image |
| 5 | **Edge Anomaly Detection** | Sharp artificial edges from paste operations |
| 6 | **Local Variance Analysis** | Smoothed/retouched areas with abnormally low variance |
| 7 | **Color Channel Anomalies** | Per-channel noise inconsistencies from editing |
| 8 | **JPEG Ghost Detection** | Double-compression artifacts revealing prior saves |
| 9 | **Metadata Forensics** | Editing software tags, date mismatches, stripped EXIF |
| 10 | **Text Region Anomaly** | Altered text in documents (names, amounts, dates) |

Each technique produces a **score from 0-100**. These are combined with weights into an **overall fraud score** with a verdict:

- **0-19**: CLEAN — No significant manipulation detected
- **20-39**: LOW — Minor anomalies detected
- **40-64**: MEDIUM — Possibly manipulated, review recommended
- **65-100**: HIGH — Very likely manipulated

## Quick Start

### Local Usage (CLI)

```bash
# Install dependencies
pip install -r requirements.txt
# Also need poppler for PDF support:
# Ubuntu/Debian: sudo apt-get install poppler-utils
# macOS: brew install poppler

# Analyze an image
python detect_fraud.py photo.jpg

# Analyze a PDF document
python detect_fraud.py invoice.pdf

# Generate a PDF report
python detect_fraud.py invoice.pdf --report fraud_report.pdf

# Export results as JSON
python detect_fraud.py receipt.png --json results.json

# Both report and JSON, custom DPI for PDF
python detect_fraud.py document.pdf --dpi 200 --report report.pdf --json out.json
```

### Google Colab Usage

1. Create a new Colab notebook
2. **Cell 1** — Install dependencies:
```python
!apt-get install -y poppler-utils > /dev/null 2>&1
!pip install -q pdf2image Pillow numpy opencv-python-headless scipy scikit-image reportlab matplotlib
print("All dependencies installed.")
```

3. **Cell 2** — Copy and paste the entire content of `fraud_detection_colab.py` (everything from the imports to the `analyze_file` function)

4. **Cell 3** — Upload and analyze:
```python
from google.colab import files

print("Upload an image (JPG/PNG) or PDF to analyze for fraud:")
uploaded = files.upload()

for filename in uploaded.keys():
    print(f"\nAnalyzing: {filename}")
    analyze_file(filename)
```

5. Run all cells. It will prompt you to upload a file and display visual results with heatmaps.

### Python API Usage

```python
from image_fraud_detector import FraudDetector, analyze_pdf, generate_report_pdf

# Single image
detector = FraudDetector(image_path="receipt.jpg")
results = detector.run_all(verbose=True)
print(f"Score: {results['overall']['score']}/100")
print(f"Verdict: {results['overall']['verdict']}")

# PDF (all pages)
all_results = analyze_pdf("invoice.pdf", dpi=300, verbose=True)

# Generate PDF report
generate_report_pdf(all_results, "fraud_report.pdf")
```

## Files

| File | Purpose |
|------|---------|
| `image_fraud_detector.py` | Core detection engine (importable module) |
| `fraud_detection_colab.py` | Google Colab version with rich visualizations |
| `detect_fraud.py` | CLI entry point |
| `requirements.txt` | Python dependencies |

## How It Works (for document fraud)

When someone edits a scanned document (e.g., changes an amount on an invoice using Photoshop):

1. **ELA** reveals the edited pixels were saved at a different compression level
2. **Noise analysis** shows the pasted text has different noise characteristics from the original scan
3. **Text region analysis** detects local contrast and micro-noise breaks around altered text
4. **Channel analysis** reveals per-channel inconsistencies where colors were manipulated
5. **Metadata** may reveal Photoshop/editing software was used

The system combines all 10 signals with weights to produce a reliable overall score.

## Requirements

- Python 3.8+
- poppler-utils (for PDF support)
- See `requirements.txt` for Python packages
