#!/usr/bin/env python3
"""
Standalone CLI for Image & Document Fraud Detection.

Usage:
    python detect_fraud.py invoice.pdf
    python detect_fraud.py photo.jpg --report output.pdf
    python detect_fraud.py scan.png --json results.json --report report.pdf
    python detect_fraud.py document.pdf --dpi 200 --report report.pdf

Supports: JPG, JPEG, PNG, BMP, TIFF, WEBP, PDF
"""

import sys
import os

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_fraud_detector import (
    FraudDetector,
    analyze_pdf,
    generate_report_pdf,
)
from PIL import Image
from pathlib import Path
import json


def score_bar(score, width=30):
    filled = int(score / 100 * width)
    if score >= 50:
        indicator = "!!"
    elif score >= 25:
        indicator = "! "
    else:
        indicator = "  "
    return f"{indicator}[{'#' * filled}{'-' * (width - filled)}] {score:.1f}/100"


def print_banner():
    print("""
  ╔══════════════════════════════════════════════════════════╗
  ║        IMAGE & DOCUMENT FRAUD DETECTOR                  ║
  ║   Detects Photoshop edits, altered text, copy-paste,    ║
  ║   double compression, metadata tampering & more         ║
  ╚══════════════════════════════════════════════════════════╝
    """)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect image and document fraud/manipulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_fraud.py invoice.pdf
  python detect_fraud.py receipt.jpg --report fraud_report.pdf
  python detect_fraud.py scan.png --json results.json
  python detect_fraud.py document.pdf --dpi 200 --report report.pdf --json out.json
        """,
    )
    parser.add_argument("input", help="Path to image (JPG/PNG/BMP/TIFF/WEBP) or PDF")
    parser.add_argument(
        "--report", "-r", default=None,
        help="Generate a PDF report at this path",
    )
    parser.add_argument(
        "--json", "-j", dest="json_out", default=None,
        help="Save detailed results as JSON",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="DPI for PDF-to-image conversion (default: 300)",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    print_banner()
    ext = Path(input_path).suffix.lower()

    if ext == ".pdf":
        print(f"  Input: {input_path} (PDF, DPI={args.dpi})")
        print(f"  Converting pages to images...")
        all_results = analyze_pdf(input_path, dpi=args.dpi, verbose=True)

    elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"):
        print(f"  Input: {input_path} ({ext.upper()} image)")
        detector = FraudDetector(image_path=input_path)
        results = detector.run_all(verbose=True)
        all_results = [(1, results)]

    else:
        print(f"Error: Unsupported file format '{ext}'")
        print("Supported: JPG, JPEG, PNG, BMP, TIFF, WEBP, PDF")
        sys.exit(1)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for page_num, res in all_results:
        ov = res["overall"]
        print(f"  Page/Image {page_num}: {ov['score']:.1f}/100 - {ov['verdict']}")
    print()

    # Generate PDF report
    if args.report:
        report_path = generate_report_pdf(all_results, args.report)
        print(f"  PDF Report saved: {report_path}")

    # JSON output
    if args.json_out:
        serializable = []
        for page, res in all_results:
            clean = {}
            for k, v in res.items():
                if isinstance(v, dict):
                    clean[k] = {
                        kk: vv for kk, vv in v.items()
                        if not isinstance(vv, Image.Image)
                        and not isinstance(vv, type(None).__mro__[0])
                    }
                else:
                    clean[k] = v
            serializable.append({"page": page, "results": clean})

        with open(args.json_out, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"  JSON Results saved: {args.json_out}")


if __name__ == "__main__":
    main()
