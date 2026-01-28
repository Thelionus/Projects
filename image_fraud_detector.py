"""
Image & Document Fraud Detection Engine
========================================
Detects manipulations (Photoshop edits, text changes, amount alterations)
in images and PDFs using multiple forensic analysis techniques.

Techniques used:
1. Error Level Analysis (ELA) - detects areas re-saved at different quality
2. Noise Analysis - inconsistent noise reveals edited regions
3. Copy-Move Detection - finds duplicated/cloned regions
4. Edge Anomaly Detection - sharp edges from paste operations
5. Local Variance Analysis - detects smoothed/retouched areas
6. Color Channel Anomalies - statistical outliers per channel
7. JPEG Ghost Detection - artifacts from double compression
8. Metadata Forensics - EXIF data inconsistencies
"""

import os
import io
import json
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
from PIL.ExifTags import TAGS
import cv2
from scipy import ndimage, fftpack
from skimage.feature import local_binary_pattern
from skimage.util import view_as_blocks


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _load_image(source):
    """Load image from path or PIL Image, return RGB PIL Image."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    return Image.open(source).convert("RGB")


def _pil_to_cv(pil_img):
    """Convert PIL RGB to OpenCV BGR numpy array."""
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _cv_to_pil(cv_img):
    """Convert OpenCV BGR to PIL RGB."""
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# 1. Error Level Analysis (ELA)
# ---------------------------------------------------------------------------

def error_level_analysis(image, quality=90, amplification=20):
    """
    Re-save the image at a given JPEG quality and measure per-pixel
    difference.  Edited regions show higher error because they were
    saved at a different compression level than the rest of the image.

    Returns:
        ela_image (PIL.Image): amplified difference image
        score (float): mean ELA intensity (0-255, higher = more suspicious)
        hot_ratio (float): fraction of pixels above the adaptive threshold
    """
    img = _load_image(image)

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")

    diff = ImageChops.difference(img, resaved)
    diff_arr = np.array(diff, dtype=np.float64)

    # Amplify
    ela_arr = np.clip(diff_arr * amplification, 0, 255).astype(np.uint8)
    ela_image = Image.fromarray(ela_arr)

    # Scoring
    gray = np.mean(diff_arr, axis=2)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    threshold = mean_val + 2.0 * std_val
    hot_ratio = float(np.sum(gray > threshold) / gray.size)

    return ela_image, mean_val, hot_ratio


def multi_quality_ela(image, qualities=(70, 80, 90, 95)):
    """
    Run ELA at multiple quality levels.  Genuine edits show consistent
    hotspots across qualities while noise does not.

    Returns list of (quality, ela_image, mean_score, hot_ratio).
    """
    results = []
    for q in qualities:
        ela_img, mean_s, hot_r = error_level_analysis(image, quality=q)
        results.append((q, ela_img, mean_s, hot_r))
    return results


# ---------------------------------------------------------------------------
# 2. Noise Analysis
# ---------------------------------------------------------------------------

def noise_analysis(image, block_size=64):
    """
    Divide image into blocks and measure noise standard deviation in each.
    Edited regions typically have different noise characteristics from the
    surrounding content.

    Returns:
        noise_map (np.ndarray): per-block noise stddev grid
        anomaly_map (np.ndarray): boolean grid marking anomalous blocks
        score (float): 0-100 anomaly score
    """
    img = _load_image(image)
    gray = np.array(img.convert("L"), dtype=np.float64)

    # High-pass filter to isolate noise
    blurred = ndimage.gaussian_filter(gray, sigma=3)
    noise = gray - blurred

    h, w = noise.shape
    bh = h // block_size
    bw = w // block_size
    noise_crop = noise[:bh * block_size, :bw * block_size]

    blocks = view_as_blocks(noise_crop, (block_size, block_size))
    noise_map = np.std(blocks, axis=(2, 3))

    overall_std = np.std(noise_map)
    overall_mean = np.mean(noise_map)
    if overall_std < 1e-6:
        anomaly_map = np.zeros_like(noise_map, dtype=bool)
        score = 0.0
    else:
        z_scores = np.abs(noise_map - overall_mean) / overall_std
        anomaly_map = z_scores > 2.0
        score = float(np.sum(anomaly_map) / anomaly_map.size * 100)

    return noise_map, anomaly_map, score


# ---------------------------------------------------------------------------
# 3. Copy-Move (Clone) Detection
# ---------------------------------------------------------------------------

def copy_move_detection(image, min_matches=10):
    """
    Detect duplicated (copy-pasted) regions using ORB feature matching.

    Returns:
        vis_image (PIL.Image): image with matched keypoints drawn
        num_matches (int): number of suspicious matches found
        score (float): 0-100 copy-move likelihood score
    """
    img = _load_image(image)
    cv_img = _pil_to_cv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=3000)
    kp, des = orb.detectAndCompute(gray, None)

    if des is None or len(kp) < 2:
        return img, 0, 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des, des, k=5)

    suspicious = []
    for group in matches:
        for m in group:
            if m.queryIdx == m.trainIdx:
                continue
            pt1 = np.array(kp[m.queryIdx].pt)
            pt2 = np.array(kp[m.trainIdx].pt)
            dist = np.linalg.norm(pt1 - pt2)
            # Close descriptors but spatially separated
            if m.distance < 30 and dist > 50:
                suspicious.append((m, dist))

    # De-duplicate symmetric pairs
    seen = set()
    unique = []
    for m, d in suspicious:
        pair = tuple(sorted([m.queryIdx, m.trainIdx]))
        if pair not in seen:
            seen.add(pair)
            unique.append(m)

    vis = cv_img.copy()
    for m in unique:
        pt1 = tuple(map(int, kp[m.queryIdx].pt))
        pt2 = tuple(map(int, kp[m.trainIdx].pt))
        cv2.line(vis, pt1, pt2, (0, 0, 255), 1)
        cv2.circle(vis, pt1, 4, (0, 255, 0), -1)
        cv2.circle(vis, pt2, 4, (0, 255, 0), -1)

    num = len(unique)
    score = min(100.0, num / max(min_matches, 1) * 100)

    return _cv_to_pil(vis), num, score


# ---------------------------------------------------------------------------
# 4. Edge Anomaly Detection
# ---------------------------------------------------------------------------

def edge_anomaly_detection(image, block_size=32):
    """
    Pasted elements often introduce sharp artificial edges that differ
    from the surrounding content.  We measure per-block edge density
    and flag statistical outliers.

    Returns:
        edge_image (PIL.Image): Canny edge visualization
        anomaly_map (np.ndarray): boolean grid of anomalous blocks
        score (float): 0-100 edge anomaly score
    """
    img = _load_image(image)
    cv_img = _pil_to_cv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Adaptive Canny
    median_val = np.median(gray)
    low = int(max(0, 0.66 * median_val))
    high = int(min(255, 1.33 * median_val))
    edges = cv2.Canny(gray, low, high)

    h, w = edges.shape
    bh, bw = h // block_size, w // block_size
    edge_crop = edges[:bh * block_size, :bw * block_size]
    blocks = view_as_blocks(edge_crop, (block_size, block_size))
    density = np.mean(blocks > 0, axis=(2, 3))

    overall_mean = np.mean(density)
    overall_std = np.std(density)
    if overall_std < 1e-6:
        anomaly_map = np.zeros_like(density, dtype=bool)
        score = 0.0
    else:
        z = np.abs(density - overall_mean) / overall_std
        anomaly_map = z > 2.5
        score = float(np.sum(anomaly_map) / anomaly_map.size * 100)

    edge_pil = Image.fromarray(edges)
    return edge_pil, anomaly_map, score


# ---------------------------------------------------------------------------
# 5. Local Variance Analysis
# ---------------------------------------------------------------------------

def local_variance_analysis(image, block_size=32):
    """
    Smoothed / retouched areas have abnormally low variance compared to
    their surroundings.

    Returns:
        variance_map (np.ndarray): per-block variance grid
        anomaly_map (np.ndarray): boolean grid
        score (float): 0-100 score
    """
    img = _load_image(image)
    gray = np.array(img.convert("L"), dtype=np.float64)

    h, w = gray.shape
    bh, bw = h // block_size, w // block_size
    crop = gray[:bh * block_size, :bw * block_size]
    blocks = view_as_blocks(crop, (block_size, block_size))
    var_map = np.var(blocks, axis=(2, 3))

    overall_mean = np.mean(var_map)
    overall_std = np.std(var_map)
    if overall_std < 1e-6:
        anomaly_map = np.zeros_like(var_map, dtype=bool)
        score = 0.0
    else:
        z = np.abs(var_map - overall_mean) / overall_std
        anomaly_map = z > 2.0
        score = float(np.sum(anomaly_map) / anomaly_map.size * 100)

    return var_map, anomaly_map, score


# ---------------------------------------------------------------------------
# 6. Color Channel Anomaly Analysis
# ---------------------------------------------------------------------------

def channel_anomaly_analysis(image, block_size=32):
    """
    In authentic images the three colour channels share similar noise
    characteristics.  Editing often affects channels unevenly.

    Returns:
        channel_diff_map (np.ndarray): per-block max inter-channel diff
        anomaly_map (np.ndarray): boolean grid
        score (float): 0-100 score
    """
    img = _load_image(image)
    arr = np.array(img, dtype=np.float64)

    h, w, _ = arr.shape
    bh, bw = h // block_size, w // block_size
    crop = arr[:bh * block_size, :bw * block_size]

    channel_stds = []
    for c in range(3):
        ch = crop[:, :, c]
        blurred = ndimage.gaussian_filter(ch, sigma=3)
        noise = ch - blurred
        blocks = view_as_blocks(noise, (block_size, block_size))
        channel_stds.append(np.std(blocks, axis=(2, 3)))

    stds = np.stack(channel_stds, axis=-1)  # (bh, bw, 3)
    diff_map = np.max(stds, axis=2) - np.min(stds, axis=2)

    overall_mean = np.mean(diff_map)
    overall_std = np.std(diff_map)
    if overall_std < 1e-6:
        anomaly_map = np.zeros(diff_map.shape, dtype=bool)
        score = 0.0
    else:
        z = np.abs(diff_map - overall_mean) / overall_std
        anomaly_map = z > 2.0
        score = float(np.sum(anomaly_map) / anomaly_map.size * 100)

    return diff_map, anomaly_map, score


# ---------------------------------------------------------------------------
# 7. JPEG Ghost Detection
# ---------------------------------------------------------------------------

def jpeg_ghost_detection(image, quality_range=(50, 95), step=5):
    """
    Re-compress at multiple quality levels and measure the residual.
    A double-compressed region will show a trough at its original quality.

    Returns:
        ghost_map (np.ndarray): per-quality mean-squared residual
        suspicious_quality (int or None): quality where a trough is detected
        score (float): 0-100 score
    """
    img = _load_image(image)
    arr = np.array(img, dtype=np.float64)

    residuals = []
    qualities = list(range(quality_range[0], quality_range[1] + 1, step))

    for q in qualities:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q)
        buf.seek(0)
        recomp = np.array(Image.open(buf).convert("RGB"), dtype=np.float64)
        mse = np.mean((arr - recomp) ** 2)
        residuals.append(mse)

    residuals = np.array(residuals)

    # Look for a local minimum (trough) in the residual curve
    suspicious_q = None
    score = 0.0
    if len(residuals) >= 3:
        # Residuals should monotonically decrease; a bump indicates double-compression
        diffs = np.diff(residuals)
        for i in range(len(diffs) - 1):
            if diffs[i] < 0 and diffs[i + 1] > 0:
                suspicious_q = qualities[i + 1]
                # Score based on trough depth
                depth = (residuals[i] + residuals[i + 2]) / 2 - residuals[i + 1]
                score = min(100.0, depth / (np.mean(residuals) + 1e-9) * 200)
                break

    ghost_map = dict(zip(qualities, residuals.tolist()))
    return ghost_map, suspicious_q, score


# ---------------------------------------------------------------------------
# 8. Metadata Forensics
# ---------------------------------------------------------------------------

def metadata_analysis(image_path):
    """
    Extract and analyse EXIF / metadata for tampering signs:
      - Editing software tags (Photoshop, GIMP, etc.)
      - Inconsistent timestamps
      - Stripped metadata (common after editing)

    Returns:
        findings (dict): structured metadata report
        score (float): 0-100 suspicion score
    """
    KNOWN_EDITORS = [
        "photoshop", "gimp", "paint.net", "affinity", "pixlr", "canva",
        "lightroom", "snapseed", "fotor", "befunky", "picmonkey",
        "adobe", "corel", "inkscape", "illustrator",
    ]

    findings = {
        "has_exif": False,
        "software": None,
        "editing_software_detected": False,
        "create_date": None,
        "modify_date": None,
        "date_mismatch": False,
        "camera_info": None,
        "raw_tags": {},
    }
    score = 0.0

    try:
        img = Image.open(image_path)
    except Exception:
        return findings, score

    exif = img.getexif()
    if not exif:
        findings["has_exif"] = False
        # Stripped EXIF is mildly suspicious for photos (not for screenshots/scans)
        score += 10
        return findings, min(100.0, score)

    findings["has_exif"] = True
    tags = {}
    for tag_id, value in exif.items():
        tag_name = TAGS.get(tag_id, str(tag_id))
        try:
            tags[tag_name] = str(value)
        except Exception:
            tags[tag_name] = repr(value)

    findings["raw_tags"] = tags

    software = tags.get("Software", "")
    findings["software"] = software
    if software:
        sw_lower = software.lower()
        for editor in KNOWN_EDITORS:
            if editor in sw_lower:
                findings["editing_software_detected"] = True
                score += 40
                break

    dt_orig = tags.get("DateTimeOriginal")
    dt_mod = tags.get("DateTime")
    findings["create_date"] = dt_orig
    findings["modify_date"] = dt_mod
    if dt_orig and dt_mod and dt_orig != dt_mod:
        findings["date_mismatch"] = True
        score += 20

    make = tags.get("Make", "")
    model = tags.get("Model", "")
    if make or model:
        findings["camera_info"] = f"{make} {model}".strip()

    return findings, min(100.0, score)


# ---------------------------------------------------------------------------
# 9. Document-Specific: Text Region Anomaly Detection
# ---------------------------------------------------------------------------

def text_region_analysis(image, block_size=16):
    """
    For scanned documents / invoices: detect localised anomalies around
    text regions that may indicate altered text (names, amounts, dates).

    Uses a combination of:
      - Local contrast inconsistency
      - Background texture discontinuity
      - Micro-noise pattern breaks

    Returns:
        heatmap (np.ndarray): anomaly heatmap
        regions (list): list of (x, y, w, h, score) suspicious rectangles
        score (float): 0-100 overall score
    """
    img = _load_image(image)
    cv_img = _pil_to_cv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    heatmap = np.zeros((h, w), dtype=np.float64)

    # --- A. Local contrast analysis ---
    local_mean = ndimage.uniform_filter(gray.astype(np.float64), size=block_size)
    local_sq_mean = ndimage.uniform_filter(gray.astype(np.float64) ** 2, size=block_size)
    local_var = np.clip(local_sq_mean - local_mean ** 2, 0, None)
    local_std = np.sqrt(local_var)

    # --- B. High-freq noise residual ---
    blurred = ndimage.gaussian_filter(gray.astype(np.float64), sigma=1.0)
    noise_res = np.abs(gray.astype(np.float64) - blurred)

    # --- C. Sliding-window anomaly ---
    stride = block_size // 2
    block_scores = []
    positions = []

    for y in range(0, h - block_size, stride):
        for x in range(0, w - block_size, stride):
            patch_std = local_std[y:y+block_size, x:x+block_size].mean()
            patch_noise = noise_res[y:y+block_size, x:x+block_size].mean()
            block_scores.append((patch_std, patch_noise))
            positions.append((x, y))

    if not block_scores:
        return heatmap, [], 0.0

    scores_arr = np.array(block_scores)
    means = scores_arr.mean(axis=0)
    stds = scores_arr.std(axis=0)
    stds[stds < 1e-6] = 1.0

    z_scores = np.abs(scores_arr - means) / stds
    combined_z = z_scores.mean(axis=1)

    for idx, (x, y) in enumerate(positions):
        heatmap[y:y+block_size, x:x+block_size] += combined_z[idx]

    # Normalize heatmap
    hm_max = heatmap.max()
    if hm_max > 0:
        heatmap = heatmap / hm_max * 255

    # Find suspicious rectangular regions
    thresh = np.percentile(heatmap, 95)
    binary = (heatmap > thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        region_score = float(heatmap[ry:ry+rh, rx:rx+rw].mean())
        if rw > 10 and rh > 5:  # filter tiny noise
            regions.append((rx, ry, rw, rh, region_score))

    regions.sort(key=lambda r: r[4], reverse=True)
    overall = float(np.mean(combined_z > 2.0) * 100)

    return heatmap, regions[:20], overall


# ---------------------------------------------------------------------------
# 10. Luminance Gradient Analysis
# ---------------------------------------------------------------------------

def luminance_gradient_analysis(image, block_size=32):
    """
    Detect unnatural luminance transitions that result from pasting
    content with different lighting conditions.

    Returns:
        gradient_map (np.ndarray): gradient magnitude image
        anomaly_map (np.ndarray): boolean grid
        score (float): 0-100 score
    """
    img = _load_image(image)
    gray = np.array(img.convert("L"), dtype=np.float64)

    # Sobel gradients
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    magnitude = np.sqrt(gx**2 + gy**2)

    h, w = magnitude.shape
    bh, bw = h // block_size, w // block_size
    crop = magnitude[:bh * block_size, :bw * block_size]
    blocks = view_as_blocks(crop, (block_size, block_size))
    grad_means = np.mean(blocks, axis=(2, 3))

    overall_mean = np.mean(grad_means)
    overall_std = np.std(grad_means)
    if overall_std < 1e-6:
        anomaly_map = np.zeros_like(grad_means, dtype=bool)
        score = 0.0
    else:
        z = np.abs(grad_means - overall_mean) / overall_std
        anomaly_map = z > 2.5
        score = float(np.sum(anomaly_map) / anomaly_map.size * 100)

    return magnitude, anomaly_map, score


# ---------------------------------------------------------------------------
# Composite Analyzer
# ---------------------------------------------------------------------------

class FraudDetector:
    """
    Run all forensic analyses and produce a consolidated report with
    an overall fraud-likelihood score.
    """

    WEIGHTS = {
        "ela": 0.20,
        "noise": 0.15,
        "copy_move": 0.15,
        "edge": 0.10,
        "local_variance": 0.10,
        "channel": 0.10,
        "jpeg_ghost": 0.05,
        "metadata": 0.05,
        "text_region": 0.05,
        "luminance": 0.05,
    }

    def __init__(self, image_path=None, pil_image=None):
        self.image_path = image_path
        if pil_image is not None:
            self.image = pil_image.convert("RGB")
        elif image_path is not None:
            self.image = Image.open(image_path).convert("RGB")
        else:
            raise ValueError("Provide image_path or pil_image")

    def run_all(self, verbose=False):
        """Run every analysis and return structured results."""
        results = {}

        # 1. ELA
        ela_img, ela_mean, ela_hot = error_level_analysis(self.image)
        ela_score = min(100, ela_hot * 1000 + ela_mean * 2)
        results["ela"] = {
            "image": ela_img,
            "mean_error": round(ela_mean, 2),
            "hot_pixel_ratio": round(ela_hot, 4),
            "score": round(ela_score, 1),
        }

        # 2. Multi-quality ELA
        mq = multi_quality_ela(self.image)
        consistent_hotspots = sum(1 for _, _, _, hr in mq if hr > 0.01)
        if consistent_hotspots >= 3:
            results["ela"]["score"] = min(100, results["ela"]["score"] + 15)
        results["ela"]["multi_quality"] = [
            {"quality": q, "mean": round(m, 2), "hot_ratio": round(h, 4)}
            for q, _, m, h in mq
        ]

        # 3. Noise
        _, noise_anom, noise_score = noise_analysis(self.image)
        results["noise"] = {"score": round(noise_score, 1)}

        # 4. Copy-Move
        cm_img, cm_matches, cm_score = copy_move_detection(self.image)
        results["copy_move"] = {
            "image": cm_img,
            "matches": cm_matches,
            "score": round(cm_score, 1),
        }

        # 5. Edge
        edge_img, _, edge_score = edge_anomaly_detection(self.image)
        results["edge"] = {"image": edge_img, "score": round(edge_score, 1)}

        # 6. Local Variance
        _, _, var_score = local_variance_analysis(self.image)
        results["local_variance"] = {"score": round(var_score, 1)}

        # 7. Channel
        _, _, ch_score = channel_anomaly_analysis(self.image)
        results["channel"] = {"score": round(ch_score, 1)}

        # 8. JPEG Ghost
        ghost_map, suspicious_q, ghost_score = jpeg_ghost_detection(self.image)
        results["jpeg_ghost"] = {
            "residual_curve": ghost_map,
            "suspicious_quality": suspicious_q,
            "score": round(ghost_score, 1),
        }

        # 9. Metadata
        if self.image_path:
            meta, meta_score = metadata_analysis(self.image_path)
        else:
            meta, meta_score = {}, 0.0
        results["metadata"] = {**meta, "score": round(meta_score, 1)}

        # 10. Text Region
        heatmap, regions, text_score = text_region_analysis(self.image)
        results["text_region"] = {
            "suspicious_regions": [
                {"x": r[0], "y": r[1], "w": r[2], "h": r[3],
                 "score": round(r[4], 1)}
                for r in regions
            ],
            "score": round(text_score, 1),
        }

        # 11. Luminance
        _, _, lum_score = luminance_gradient_analysis(self.image)
        results["luminance"] = {"score": round(lum_score, 1)}

        # --- Weighted Overall Score ---
        weighted = 0.0
        for key, weight in self.WEIGHTS.items():
            weighted += results.get(key, {}).get("score", 0) * weight
        overall = min(100.0, weighted)

        # Verdict
        if overall >= 65:
            verdict = "HIGH - Very likely manipulated"
        elif overall >= 40:
            verdict = "MEDIUM - Possibly manipulated, review recommended"
        elif overall >= 20:
            verdict = "LOW - Minor anomalies detected"
        else:
            verdict = "CLEAN - No significant manipulation detected"

        results["overall"] = {
            "score": round(overall, 1),
            "verdict": verdict,
        }

        if verbose:
            self._print_report(results)

        return results

    @staticmethod
    def _print_report(results):
        print("\n" + "=" * 60)
        print("   IMAGE FRAUD DETECTION REPORT")
        print("=" * 60)

        order = [
            ("ela", "Error Level Analysis (ELA)"),
            ("noise", "Noise Consistency"),
            ("copy_move", "Copy-Move Detection"),
            ("edge", "Edge Anomalies"),
            ("local_variance", "Local Variance"),
            ("channel", "Color Channel Anomalies"),
            ("jpeg_ghost", "JPEG Ghost / Double Compression"),
            ("metadata", "Metadata Forensics"),
            ("text_region", "Text Region Anomalies"),
            ("luminance", "Luminance Gradient"),
        ]

        for key, label in order:
            r = results.get(key, {})
            score = r.get("score", 0)
            bar = _score_bar(score)
            print(f"\n  {label}")
            print(f"    Score: {score:5.1f}/100  {bar}")

            if key == "ela":
                print(f"    Mean Error: {r.get('mean_error', 'N/A')}")
                print(f"    Hot Pixel Ratio: {r.get('hot_pixel_ratio', 'N/A')}")
            elif key == "copy_move":
                print(f"    Suspicious Matches: {r.get('matches', 0)}")
            elif key == "jpeg_ghost":
                sq = r.get("suspicious_quality")
                print(f"    Suspicious Quality: {sq if sq else 'None detected'}")
            elif key == "metadata":
                if r.get("editing_software_detected"):
                    print(f"    *** Editing Software: {r.get('software')} ***")
                if r.get("date_mismatch"):
                    print("    *** Date mismatch detected ***")
            elif key == "text_region":
                regions = r.get("suspicious_regions", [])
                if regions:
                    print(f"    Suspicious Regions: {len(regions)}")
                    for i, reg in enumerate(regions[:5]):
                        print(f"      #{i+1}: ({reg['x']},{reg['y']}) "
                              f"{reg['w']}x{reg['h']} score={reg['score']}")

        ov = results["overall"]
        print(f"\n{'=' * 60}")
        print(f"  OVERALL SCORE: {ov['score']:.1f} / 100")
        print(f"  VERDICT: {ov['verdict']}")
        print(f"{'=' * 60}\n")


def _score_bar(score, width=20):
    filled = int(score / 100 * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ---------------------------------------------------------------------------
# PDF Support
# ---------------------------------------------------------------------------

def analyze_pdf(pdf_path, dpi=300, verbose=True):
    """
    Convert each page of a PDF to an image and run full fraud detection.

    Requires poppler (pdf2image).

    Returns:
        list of (page_number, results_dict)
    """
    from pdf2image import convert_from_path

    pages = convert_from_path(pdf_path, dpi=dpi)
    all_results = []

    for i, page in enumerate(pages):
        if verbose:
            print(f"\n{'*' * 60}")
            print(f"  ANALYZING PAGE {i + 1} of {len(pages)}")
            print(f"{'*' * 60}")

        detector = FraudDetector(pil_image=page)
        results = detector.run_all(verbose=verbose)
        all_results.append((i + 1, results))

    return all_results


# ---------------------------------------------------------------------------
# Generate PDF Report
# ---------------------------------------------------------------------------

def generate_report_pdf(results_list, output_path="fraud_report.pdf"):
    """
    Generate a PDF report from analysis results.

    Args:
        results_list: list of (page_number, results) or single results dict
        output_path: output PDF file path
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"], fontSize=18,
        spaceAfter=20, textColor=colors.HexColor("#1a1a2e")
    )
    heading_style = ParagraphStyle(
        "CustomHeading", parent=styles["Heading2"], fontSize=13,
        spaceAfter=8, textColor=colors.HexColor("#16213e")
    )

    story.append(Paragraph("Image Fraud Detection Report", title_style))
    story.append(Spacer(1, 12))

    # Normalize input
    if isinstance(results_list, dict):
        results_list = [(1, results_list)]

    for page_num, results in results_list:
        story.append(Paragraph(f"Page / Image {page_num}", heading_style))

        overall = results.get("overall", {})
        score = overall.get("score", 0)
        verdict = overall.get("verdict", "N/A")

        # Verdict color
        if score >= 65:
            v_color = colors.HexColor("#e74c3c")
        elif score >= 40:
            v_color = colors.HexColor("#f39c12")
        else:
            v_color = colors.HexColor("#27ae60")

        story.append(Paragraph(
            f'<font color="{v_color}"><b>Overall Score: {score:.1f}/100 '
            f'&mdash; {verdict}</b></font>', styles["Normal"]
        ))
        story.append(Spacer(1, 8))

        # Detail table
        rows = [["Analysis", "Score", "Details"]]
        detail_map = {
            "ela": ("Error Level Analysis", lambda r: f"Mean err={r.get('mean_error','?')}"),
            "noise": ("Noise Consistency", lambda r: ""),
            "copy_move": ("Copy-Move Detection", lambda r: f"{r.get('matches',0)} matches"),
            "edge": ("Edge Anomalies", lambda r: ""),
            "local_variance": ("Local Variance", lambda r: ""),
            "channel": ("Channel Anomalies", lambda r: ""),
            "jpeg_ghost": ("JPEG Ghost", lambda r: f"Q={r.get('suspicious_quality','None')}"),
            "metadata": ("Metadata", lambda r:
                          f"SW={r.get('software','?')}" if r.get('editing_software_detected') else ""),
            "text_region": ("Text Regions", lambda r:
                             f"{len(r.get('suspicious_regions',[]))} regions"),
            "luminance": ("Luminance Gradient", lambda r: ""),
        }

        for key, (label, detail_fn) in detail_map.items():
            r = results.get(key, {})
            s = r.get("score", 0)
            rows.append([label, f"{s:.1f}", detail_fn(r)])

        table = Table(rows, colWidths=[2.5*inch, 1*inch, 3*inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#f8f9fa"), colors.white]),
        ]))
        story.append(table)
        story.append(Spacer(1, 20))

        # Suspicious text regions
        text_r = results.get("text_region", {})
        regions = text_r.get("suspicious_regions", [])
        if regions:
            story.append(Paragraph("Suspicious Text Regions:", heading_style))
            for idx, reg in enumerate(regions[:10]):
                story.append(Paragraph(
                    f"&bull; Region #{idx+1}: position=({reg['x']},{reg['y']}) "
                    f"size={reg['w']}x{reg['h']} score={reg['score']}",
                    styles["Normal"]
                ))
            story.append(Spacer(1, 12))

    doc.build(story)
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point (when run directly)
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Image & Document Fraud Detection Tool"
    )
    parser.add_argument("input", help="Path to image or PDF file")
    parser.add_argument("--output", "-o", default=None,
                        help="Path for PDF report output")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for PDF conversion (default: 300)")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="Path for JSON results output")
    args = parser.parse_args()

    input_path = args.input
    ext = Path(input_path).suffix.lower()

    if ext == ".pdf":
        all_results = analyze_pdf(input_path, dpi=args.dpi, verbose=True)
    elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
        print(f"\nAnalyzing image: {input_path}")
        detector = FraudDetector(image_path=input_path)
        results = detector.run_all(verbose=True)
        all_results = [(1, results)]
    else:
        print(f"Unsupported file format: {ext}")
        return

    # Generate PDF report
    if args.output:
        report_path = generate_report_pdf(all_results, args.output)
        print(f"\nPDF report saved to: {report_path}")

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
                    }
                else:
                    clean[k] = v
            serializable.append({"page": page, "results": clean})
        with open(args.json_out, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"JSON results saved to: {args.json_out}")


if __name__ == "__main__":
    main()
