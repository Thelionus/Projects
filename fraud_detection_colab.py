"""
============================================================
  IMAGE & DOCUMENT FRAUD DETECTION - Google Colab Version
============================================================

HOW TO USE IN GOOGLE COLAB:
  1. Create a new Colab notebook
  2. In Cell 1, paste the INSTALL block below
  3. In Cell 2, paste the FULL DETECTION CODE block
  4. In Cell 3, paste the RUN block
  5. Run all cells - it will prompt you to upload a file

Detects: Photoshop edits, altered text/names/amounts,
         copy-paste regions, double compression, and more.
============================================================
"""

# =============================================
# CELL 1: INSTALL DEPENDENCIES
# =============================================
# Paste this into the first Colab cell:
"""
!apt-get install -y poppler-utils > /dev/null 2>&1
!pip install -q pdf2image Pillow numpy opencv-python-headless scipy scikit-image reportlab matplotlib
print("All dependencies installed.")
"""

# =============================================
# CELL 2: FULL DETECTION CODE
# =============================================
# Paste everything below (from the imports to the end) into the second Colab cell:

import os
import io
import tempfile
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageDraw, ImageFont
from PIL.ExifTags import TAGS
import cv2
from scipy import ndimage
from skimage.util import view_as_blocks
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display, HTML

# ---- Custom Heatmap Colormap ----
FRAUD_CMAP = LinearSegmentedColormap.from_list(
    "fraud", ["#000000", "#1a0a3e", "#5c1a8e", "#c62828", "#ff6f00", "#ffeb3b", "#ffffff"]
)


# ============================================================
#  ANALYSIS FUNCTIONS
# ============================================================

def error_level_analysis(img, quality=90, amplification=20):
    """ELA: re-save at given quality, measure pixel difference."""
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")

    diff = ImageChops.difference(img, resaved)
    diff_arr = np.array(diff, dtype=np.float64)
    ela_arr = np.clip(diff_arr * amplification, 0, 255).astype(np.uint8)

    gray = np.mean(diff_arr, axis=2)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    threshold = mean_val + 2.0 * std_val
    hot_ratio = float(np.sum(gray > threshold) / gray.size)

    return Image.fromarray(ela_arr), mean_val, hot_ratio


def multi_quality_ela(img, qualities=(70, 80, 90, 95)):
    """Run ELA at multiple quality levels for consistency check."""
    results = []
    for q in qualities:
        _, mean_s, hot_r = error_level_analysis(img, quality=q)
        results.append((q, mean_s, hot_r))
    return results


def noise_analysis(img, block_size=64):
    """Detect inconsistent noise patterns across blocks."""
    gray = np.array(img.convert("L"), dtype=np.float64)
    blurred = ndimage.gaussian_filter(gray, sigma=3)
    noise = gray - blurred

    h, w = noise.shape
    bh, bw = h // block_size, w // block_size
    if bh == 0 or bw == 0:
        return np.zeros((1, 1)), np.zeros((1, 1), dtype=bool), 0.0

    noise_crop = noise[:bh * block_size, :bw * block_size]
    blocks = view_as_blocks(noise_crop, (block_size, block_size))
    noise_map = np.std(blocks, axis=(2, 3))

    overall_std = np.std(noise_map)
    overall_mean = np.mean(noise_map)
    if overall_std < 1e-6:
        return noise_map, np.zeros_like(noise_map, dtype=bool), 0.0

    z_scores = np.abs(noise_map - overall_mean) / overall_std
    anomaly_map = z_scores > 2.0
    score = float(np.sum(anomaly_map) / anomaly_map.size * 100)
    return noise_map, anomaly_map, score


def copy_move_detection(img, min_matches=10):
    """Detect duplicated (copy-pasted) regions using ORB features."""
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
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
            if m.distance < 30 and dist > 50:
                suspicious.append((m, dist))

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

    vis_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    num = len(unique)
    score = min(100.0, num / max(min_matches, 1) * 100)
    return vis_pil, num, score


def edge_anomaly_detection(img, block_size=32):
    """Detect sharp artificial edges from paste operations."""
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    median_val = np.median(gray)
    low = int(max(0, 0.66 * median_val))
    high = int(min(255, 1.33 * median_val))
    edges = cv2.Canny(gray, low, high)

    h, w = edges.shape
    bh, bw = h // block_size, w // block_size
    if bh == 0 or bw == 0:
        return Image.fromarray(edges), np.zeros((1, 1), dtype=bool), 0.0

    edge_crop = edges[:bh * block_size, :bw * block_size]
    blocks = view_as_blocks(edge_crop, (block_size, block_size))
    density = np.mean(blocks > 0, axis=(2, 3))

    overall_mean = np.mean(density)
    overall_std = np.std(density)
    if overall_std < 1e-6:
        return Image.fromarray(edges), np.zeros_like(density, dtype=bool), 0.0

    z = np.abs(density - overall_mean) / overall_std
    anomaly_map = z > 2.5
    score = float(np.sum(anomaly_map) / anomaly_map.size * 100)
    return Image.fromarray(edges), anomaly_map, score


def local_variance_analysis(img, block_size=32):
    """Detect smoothed/retouched areas with abnormally low variance."""
    gray = np.array(img.convert("L"), dtype=np.float64)

    h, w = gray.shape
    bh, bw = h // block_size, w // block_size
    if bh == 0 or bw == 0:
        return np.zeros((1, 1)), np.zeros((1, 1), dtype=bool), 0.0

    crop = gray[:bh * block_size, :bw * block_size]
    blocks = view_as_blocks(crop, (block_size, block_size))
    var_map = np.var(blocks, axis=(2, 3))

    overall_mean = np.mean(var_map)
    overall_std = np.std(var_map)
    if overall_std < 1e-6:
        return var_map, np.zeros_like(var_map, dtype=bool), 0.0

    z = np.abs(var_map - overall_mean) / overall_std
    anomaly_map = z > 2.0
    score = float(np.sum(anomaly_map) / anomaly_map.size * 100)
    return var_map, anomaly_map, score


def channel_anomaly_analysis(img, block_size=32):
    """Detect per-channel noise inconsistencies from editing."""
    arr = np.array(img, dtype=np.float64)
    h, w, _ = arr.shape
    bh, bw = h // block_size, w // block_size
    if bh == 0 or bw == 0:
        return np.zeros((1, 1)), np.zeros((1, 1), dtype=bool), 0.0

    crop = arr[:bh * block_size, :bw * block_size]

    channel_stds = []
    for c in range(3):
        ch = crop[:, :, c]
        blurred = ndimage.gaussian_filter(ch, sigma=3)
        noise = ch - blurred
        blocks = view_as_blocks(noise, (block_size, block_size))
        channel_stds.append(np.std(blocks, axis=(2, 3)))

    stds = np.stack(channel_stds, axis=-1)
    diff_map = np.max(stds, axis=2) - np.min(stds, axis=2)

    overall_mean = np.mean(diff_map)
    overall_std = np.std(diff_map)
    if overall_std < 1e-6:
        return diff_map, np.zeros(diff_map.shape, dtype=bool), 0.0

    z = np.abs(diff_map - overall_mean) / overall_std
    anomaly_map = z > 2.0
    score = float(np.sum(anomaly_map) / anomaly_map.size * 100)
    return diff_map, anomaly_map, score


def jpeg_ghost_detection(img, quality_range=(50, 95), step=5):
    """Detect double-compression artifacts via residual curve analysis."""
    arr = np.array(img, dtype=np.float64)
    qualities = list(range(quality_range[0], quality_range[1] + 1, step))
    residuals = []

    for q in qualities:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q)
        buf.seek(0)
        recomp = np.array(Image.open(buf).convert("RGB"), dtype=np.float64)
        mse = np.mean((arr - recomp) ** 2)
        residuals.append(mse)

    residuals = np.array(residuals)
    suspicious_q = None
    score = 0.0

    if len(residuals) >= 3:
        diffs = np.diff(residuals)
        for i in range(len(diffs) - 1):
            if diffs[i] < 0 and diffs[i + 1] > 0:
                suspicious_q = qualities[i + 1]
                depth = (residuals[i] + residuals[i + 2]) / 2 - residuals[i + 1]
                score = min(100.0, depth / (np.mean(residuals) + 1e-9) * 200)
                break

    return dict(zip(qualities, residuals.tolist())), suspicious_q, score


def metadata_analysis(image_path):
    """Check EXIF for editing software, date mismatches, etc."""
    KNOWN_EDITORS = [
        "photoshop", "gimp", "paint.net", "affinity", "pixlr", "canva",
        "lightroom", "snapseed", "fotor", "befunky", "picmonkey",
        "adobe", "corel", "inkscape", "illustrator",
    ]
    findings = {
        "has_exif": False, "software": None,
        "editing_software_detected": False,
        "date_mismatch": False, "camera_info": None,
    }
    score = 0.0

    try:
        exif_img = Image.open(image_path)
        exif = exif_img.getexif()
    except Exception:
        return findings, score

    if not exif:
        findings["has_exif"] = False
        score += 10
        return findings, min(100.0, score)

    findings["has_exif"] = True
    tags = {}
    for tag_id, value in exif.items():
        tag_name = TAGS.get(tag_id, str(tag_id))
        try:
            tags[tag_name] = str(value)
        except Exception:
            pass

    software = tags.get("Software", "")
    findings["software"] = software
    if software:
        for editor in KNOWN_EDITORS:
            if editor in software.lower():
                findings["editing_software_detected"] = True
                score += 40
                break

    dt_orig = tags.get("DateTimeOriginal")
    dt_mod = tags.get("DateTime")
    if dt_orig and dt_mod and dt_orig != dt_mod:
        findings["date_mismatch"] = True
        score += 20

    make = tags.get("Make", "")
    model = tags.get("Model", "")
    if make or model:
        findings["camera_info"] = f"{make} {model}".strip()

    return findings, min(100.0, score)


def text_region_analysis(img, block_size=16):
    """
    Detect altered text regions (names, amounts, dates) in documents.
    Uses local contrast + micro-noise analysis.
    """
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    heatmap = np.zeros((h, w), dtype=np.float64)

    local_mean = ndimage.uniform_filter(gray.astype(np.float64), size=block_size)
    local_sq_mean = ndimage.uniform_filter(gray.astype(np.float64) ** 2, size=block_size)
    local_var = np.clip(local_sq_mean - local_mean ** 2, 0, None)
    local_std = np.sqrt(local_var)

    blurred = ndimage.gaussian_filter(gray.astype(np.float64), sigma=1.0)
    noise_res = np.abs(gray.astype(np.float64) - blurred)

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

    hm_max = heatmap.max()
    if hm_max > 0:
        heatmap = heatmap / hm_max * 255

    thresh = np.percentile(heatmap, 95)
    binary = (heatmap > thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        region_score = float(heatmap[ry:ry+rh, rx:rx+rw].mean())
        if rw > 10 and rh > 5:
            regions.append((rx, ry, rw, rh, region_score))

    regions.sort(key=lambda r: r[4], reverse=True)
    overall = float(np.mean(combined_z > 2.0) * 100)
    return heatmap, regions[:20], overall


def luminance_gradient_analysis(img, block_size=32):
    """Detect unnatural luminance transitions from pasting."""
    gray = np.array(img.convert("L"), dtype=np.float64)
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    magnitude = np.sqrt(gx**2 + gy**2)

    h, w = magnitude.shape
    bh, bw = h // block_size, w // block_size
    if bh == 0 or bw == 0:
        return magnitude, np.zeros((1, 1), dtype=bool), 0.0

    crop = magnitude[:bh * block_size, :bw * block_size]
    blocks = view_as_blocks(crop, (block_size, block_size))
    grad_means = np.mean(blocks, axis=(2, 3))

    overall_mean = np.mean(grad_means)
    overall_std = np.std(grad_means)
    if overall_std < 1e-6:
        return magnitude, np.zeros_like(grad_means, dtype=bool), 0.0

    z = np.abs(grad_means - overall_mean) / overall_std
    anomaly_map = z > 2.5
    score = float(np.sum(anomaly_map) / anomaly_map.size * 100)
    return magnitude, anomaly_map, score


# ============================================================
#  DOCUMENT-SPECIFIC FRAUD DETECTION
# ============================================================

def header_name_analysis(img):
    """
    Specifically analyze the header/name area of documents.
    Names on pay stubs, invoices typically appear in top-left.
    Compare this region to the rest of the document.
    AGGRESSIVE detection for image-edited names.
    """
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape

    # Define header region (top 20% of document, left 60%) - EXPANDED
    header_h = int(h * 0.20)
    header_w = int(w * 0.60)
    header_region = gray[:header_h, :header_w]

    # Define body region (middle section for cleaner comparison)
    body_start = int(h * 0.25)
    body_end = int(h * 0.75)
    body_region = gray[body_start:body_end, :]

    findings = []
    score = 0.0

    # 1. Compare noise levels - MORE SENSITIVE
    header_blur = ndimage.gaussian_filter(header_region, sigma=0.8)
    header_noise = np.std(header_region - header_blur)

    body_blur = ndimage.gaussian_filter(body_region, sigma=0.8)
    body_noise = np.std(body_region - body_blur)

    noise_diff = abs(header_noise - body_noise)
    noise_ratio = noise_diff / (body_noise + 0.001)
    if noise_ratio > 0.15:  # 15% difference (was 30%)
        findings.append(f"Header noise differs by {noise_ratio*100:.1f}% (threshold: 15%)")
        score += 30

    # 2. Analyze text rendering in header specifically
    header_gray = gray[:header_h, :header_w].astype(np.uint8)

    # Find text in header using adaptive threshold
    header_thresh = cv2.adaptiveThreshold(
        header_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find contours (text regions) in header
    contours, _ = cv2.findContours(header_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    header_text_features = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw > 5 and ch > 3 and cw * ch > 50:  # Lower threshold to catch more text
            roi = header_gray[y:y+ch, x:x+cw]
            if roi.size == 0:
                continue

            # Features
            edges = cv2.Canny(roi, 30, 100)  # More sensitive edge detection
            edge_density = np.mean(edges > 0)

            # Background around text - EXPANDED padding
            pad = 5
            y1, y2 = max(0, y-pad), min(header_h, y+ch+pad)
            x1, x2 = max(0, x-pad), min(header_w, x+cw+pad)
            bg_region = header_gray[y1:y2, x1:x2]
            bg_std = np.std(bg_region)
            bg_mean = np.mean(bg_region)

            # High-frequency noise in the region
            roi_float = roi.astype(np.float64)
            roi_blur = ndimage.gaussian_filter(roi_float, sigma=0.5)
            roi_noise = np.std(roi_float - roi_blur)

            header_text_features.append({
                'bbox': (x, y, cw, ch),
                'edge_density': edge_density,
                'bg_std': bg_std,
                'bg_mean': bg_mean,
                'mean_val': np.mean(roi),
                'noise_level': roi_noise,
            })

    # 3. Compare header text to body text
    body_gray = gray[body_start:body_end, :].astype(np.uint8)
    body_thresh = cv2.adaptiveThreshold(
        body_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    body_contours, _ = cv2.findContours(body_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    body_text_features = []
    for cnt in body_contours[:100]:  # Sample more
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw > 5 and ch > 3 and cw * ch > 50:
            roi = body_gray[y:y+ch, x:x+cw]
            if roi.size == 0:
                continue
            edges = cv2.Canny(roi, 30, 100)
            edge_density = np.mean(edges > 0)

            roi_float = roi.astype(np.float64)
            roi_blur = ndimage.gaussian_filter(roi_float, sigma=0.5)
            roi_noise = np.std(roi_float - roi_blur)

            body_text_features.append({
                'edge_density': edge_density,
                'mean_val': np.mean(roi),
                'noise_level': roi_noise,
            })

    # Compare features - MORE SENSITIVE thresholds
    if header_text_features and body_text_features:
        header_edge_avg = np.mean([f['edge_density'] for f in header_text_features])
        body_edge_avg = np.mean([f['edge_density'] for f in body_text_features])

        edge_diff = abs(header_edge_avg - body_edge_avg)
        if edge_diff > 0.05:  # Was 0.1
            findings.append(f"Header text sharpness: {header_edge_avg:.3f} vs body: {body_edge_avg:.3f}")
            score += 25

        # Compare mean intensities
        header_intensity_avg = np.mean([f['mean_val'] for f in header_text_features])
        body_intensity_avg = np.mean([f['mean_val'] for f in body_text_features])

        intensity_diff = abs(header_intensity_avg - body_intensity_avg)
        if intensity_diff > 10:  # Was 20
            findings.append(f"Header text darkness: {header_intensity_avg:.1f} vs body: {body_intensity_avg:.1f}")
            score += 20

        # NEW: Compare noise levels in text regions
        header_noise_avg = np.mean([f['noise_level'] for f in header_text_features])
        body_noise_avg = np.mean([f['noise_level'] for f in body_text_features])

        text_noise_diff = abs(header_noise_avg - body_noise_avg)
        if text_noise_diff > 1.0:
            findings.append(f"Header text noise: {header_noise_avg:.2f} vs body: {body_noise_avg:.2f}")
            score += 25

    # 4. Check for rectangular "patch" patterns - MORE SENSITIVE
    local_var = ndimage.generic_filter(header_gray.astype(np.float64), np.var, size=8)
    very_uniform = local_var < 10  # Was 5

    uniform_ratio = np.mean(very_uniform)
    if uniform_ratio > 0.05 and uniform_ratio < 0.95:  # Was 0.1 to 0.9
        findings.append(f"Uniform patches in header: {uniform_ratio*100:.1f}%")
        score += 15

    # 5. Analyze specific "name-like" regions - MORE AGGRESSIVE
    name_candidates = [f for f in header_text_features if f['bbox'][2] > 30 and f['bbox'][3] > 8]
    for nc in name_candidates:
        bg_std = nc['bg_std']
        if bg_std < 8:  # Was 3 - more sensitive
            findings.append(f"Name at ({nc['bbox'][0]},{nc['bbox'][1]}): suspiciously clean background (std={bg_std:.1f})")
            score += 20

    # 6. NEW: ELA-like analysis on header specifically
    header_pil = Image.fromarray(header_gray)
    buf = io.BytesIO()
    header_pil.save(buf, "JPEG", quality=85)
    buf.seek(0)
    header_resaved = np.array(Image.open(buf).convert("L"), dtype=np.float64)

    ela_diff = np.abs(header_gray.astype(np.float64) - header_resaved)
    ela_mean = np.mean(ela_diff)
    ela_std = np.std(ela_diff)

    # Compare ELA to body
    body_pil = Image.fromarray(body_gray)
    buf2 = io.BytesIO()
    body_pil.save(buf2, "JPEG", quality=85)
    buf2.seek(0)
    body_resaved = np.array(Image.open(buf2).convert("L"), dtype=np.float64)

    body_ela_diff = np.abs(body_gray.astype(np.float64) - body_resaved)
    body_ela_mean = np.mean(body_ela_diff)

    ela_ratio = ela_mean / (body_ela_mean + 0.001)
    if ela_ratio > 1.2 or ela_ratio < 0.8:  # 20% difference
        findings.append(f"Header ELA differs: {ela_mean:.2f} vs body: {body_ela_mean:.2f} (ratio: {ela_ratio:.2f})")
        score += 25

    # 7. NEW: Check for anti-aliasing inconsistencies
    # Edited text often has different anti-aliasing than original
    header_edges = cv2.Canny(header_gray, 30, 100)
    edge_pixels = header_gray[header_edges > 0]
    if len(edge_pixels) > 10:
        edge_variation = np.std(edge_pixels)
        if edge_variation < 20:  # Too uniform = digitally rendered text
            findings.append(f"Header text anti-aliasing too uniform (std={edge_variation:.1f})")
            score += 15

    return findings, min(100, score), header_text_features


def detect_text_regions(img):
    """Use edge detection and morphology to find text-like regions."""
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold to find text
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Dilate to connect text characters into regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter by aspect ratio and size (text-like regions)
        if w > 20 and h > 5 and w/h > 1.5 and h < 100:
            text_regions.append((x, y, w, h))

    return text_regions, gray, thresh


def font_consistency_analysis(img):
    """
    Analyze font rendering consistency across text regions.
    Different fonts or rendering = potential text replacement.
    """
    text_regions, gray, thresh = detect_text_regions(img)

    if len(text_regions) < 3:
        return [], 0.0, None

    region_features = []

    for (x, y, w, h) in text_regions:
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        # Extract features that characterize text rendering
        # 1. Edge density (how sharp are the letters)
        edges = cv2.Canny(roi, 50, 150)
        edge_density = np.mean(edges > 0)

        # 2. Stroke width estimation via distance transform
        roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        dist = cv2.distanceTransform(roi_bin, cv2.DIST_L2, 5)
        stroke_width = np.mean(dist[dist > 0]) if np.any(dist > 0) else 0

        # 3. Local contrast
        local_std = np.std(roi)

        # 4. Mean intensity
        mean_intensity = np.mean(roi)

        # 5. Gradient magnitude (text sharpness)
        gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.mean(np.sqrt(gx**2 + gy**2))

        region_features.append({
            'bbox': (x, y, w, h),
            'edge_density': edge_density,
            'stroke_width': stroke_width,
            'local_std': local_std,
            'mean_intensity': mean_intensity,
            'grad_mag': grad_mag,
        })

    if len(region_features) < 3:
        return [], 0.0, None

    # Convert to numpy for statistical analysis
    features_matrix = np.array([
        [f['edge_density'], f['stroke_width'], f['local_std'], f['grad_mag']]
        for f in region_features
    ])

    # Normalize features
    means = features_matrix.mean(axis=0)
    stds = features_matrix.std(axis=0)
    stds[stds < 1e-6] = 1.0
    z_scores = np.abs(features_matrix - means) / stds

    # Combined anomaly score per region
    region_anomaly_scores = z_scores.mean(axis=1)

    # Flag outliers (z > 1.5 is suspicious for documents)
    anomalies = []
    for i, f in enumerate(region_features):
        f['anomaly_score'] = float(region_anomaly_scores[i])
        if region_anomaly_scores[i] > 1.5:
            anomalies.append(f)

    # Sort by anomaly score
    anomalies.sort(key=lambda x: x['anomaly_score'], reverse=True)

    # Overall score based on how many anomalous regions
    overall_score = min(100, len(anomalies) / max(len(region_features), 1) * 200)

    return anomalies, overall_score, region_features


def background_uniformity_analysis(img):
    """
    Check if background around/under text is uniform.
    Edited text often has slightly different background.
    """
    text_regions, gray, _ = detect_text_regions(img)

    if len(text_regions) < 2:
        return [], 0.0

    background_samples = []

    for (x, y, w, h) in text_regions:
        # Sample background just above and below text
        pad = 3

        # Above
        if y > pad:
            above = gray[max(0,y-pad-2):y-pad, x:x+w]
            if above.size > 0:
                background_samples.append({
                    'bbox': (x, y, w, h),
                    'location': 'above',
                    'mean': float(np.mean(above)),
                    'std': float(np.std(above)),
                    'median': float(np.median(above)),
                })

        # Below
        if y + h + pad < gray.shape[0]:
            below = gray[y+h+pad:min(gray.shape[0], y+h+pad+2), x:x+w]
            if below.size > 0:
                background_samples.append({
                    'bbox': (x, y, w, h),
                    'location': 'below',
                    'mean': float(np.mean(below)),
                    'std': float(np.std(below)),
                    'median': float(np.median(below)),
                })

    if len(background_samples) < 3:
        return [], 0.0

    # Analyze consistency
    means = np.array([s['mean'] for s in background_samples])
    overall_mean = np.mean(means)
    overall_std = np.std(means)

    if overall_std < 1e-6:
        return [], 0.0

    # Find outliers
    anomalies = []
    for s in background_samples:
        z = abs(s['mean'] - overall_mean) / overall_std
        s['z_score'] = float(z)
        if z > 2.0:
            anomalies.append(s)

    score = min(100, len(anomalies) / max(len(background_samples), 1) * 300)
    return anomalies, score


def text_alignment_analysis(img):
    """
    Check if text is properly aligned on baselines.
    Pasted text often has slight vertical misalignment.
    """
    text_regions, gray, _ = detect_text_regions(img)

    if len(text_regions) < 3:
        return [], 0.0

    # Group regions by approximate y-position (same line)
    sorted_regions = sorted(text_regions, key=lambda r: r[1])

    lines = []
    current_line = [sorted_regions[0]]

    for region in sorted_regions[1:]:
        # If y is within tolerance, same line
        if abs(region[1] - current_line[-1][1]) < 10:
            current_line.append(region)
        else:
            if len(current_line) >= 2:
                lines.append(current_line)
            current_line = [region]

    if len(current_line) >= 2:
        lines.append(current_line)

    # Analyze alignment within each line
    misalignments = []

    for line in lines:
        if len(line) < 2:
            continue

        # Get baseline (bottom of each region)
        baselines = [r[1] + r[3] for r in line]
        mean_baseline = np.mean(baselines)

        for i, region in enumerate(line):
            deviation = abs(baselines[i] - mean_baseline)
            if deviation > 3:  # More than 3 pixels off
                misalignments.append({
                    'bbox': region,
                    'baseline_deviation': float(deviation),
                    'expected_baseline': float(mean_baseline),
                    'actual_baseline': float(baselines[i]),
                })

    score = min(100, len(misalignments) * 15)
    return misalignments, score


def micro_pattern_analysis(img, block_size=8):
    """
    Analyze micro-level patterns around text.
    Detects subtle differences in rendering, anti-aliasing, or compression.
    """
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Compute local binary pattern-like features
    h, w = gray.shape

    # High-frequency component (very fine details)
    blurred = ndimage.gaussian_filter(gray, sigma=0.5)
    high_freq = gray - blurred

    # Compute local statistics in small blocks
    bh, bw = h // block_size, w // block_size
    if bh == 0 or bw == 0:
        return np.zeros((1,1)), [], 0.0

    crop = high_freq[:bh*block_size, :bw*block_size]
    blocks = view_as_blocks(crop, (block_size, block_size))

    # Features per block
    block_energy = np.sum(blocks**2, axis=(2, 3))
    block_mean = np.mean(blocks, axis=(2, 3))
    block_std = np.std(blocks, axis=(2, 3))

    # Combined feature
    feature_map = block_energy / (block_std + 1e-6)

    # Find anomalies
    overall_mean = np.mean(feature_map)
    overall_std = np.std(feature_map)

    if overall_std < 1e-6:
        return feature_map, [], 0.0

    z_map = np.abs(feature_map - overall_mean) / overall_std
    anomaly_mask = z_map > 2.5

    # Get coordinates of anomalies
    anomaly_coords = []
    for i in range(anomaly_mask.shape[0]):
        for j in range(anomaly_mask.shape[1]):
            if anomaly_mask[i, j]:
                anomaly_coords.append({
                    'block': (j * block_size, i * block_size),
                    'size': block_size,
                    'z_score': float(z_map[i, j]),
                })

    score = min(100, np.sum(anomaly_mask) / anomaly_mask.size * 200)
    return feature_map, anomaly_coords, score


def is_document_image(img):
    """Detect if image is a document (vs a photo)."""
    gray = np.array(img.convert("L"))

    # Documents tend to have:
    # 1. High contrast (mostly white background with dark text)
    # 2. Bimodal histogram
    # 3. Lots of straight edges

    # Check histogram bimodality
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    hist = hist / hist.sum()

    # Documents have peaks near 0 (text) and 255 (background)
    dark_ratio = hist[:50].sum()
    light_ratio = hist[200:].sum()

    is_doc = light_ratio > 0.5 and dark_ratio > 0.01
    confidence = min(1.0, light_ratio + dark_ratio * 2)

    return is_doc, confidence


# ============================================================
#  COMPOSITE FRAUD DETECTOR
# ============================================================

# Weights for PHOTO analysis
PHOTO_WEIGHTS = {
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

# Weights for DOCUMENT analysis (pay stubs, invoices, etc.)
DOCUMENT_WEIGHTS = {
    "ela": 0.08,
    "noise": 0.05,
    "copy_move": 0.00,  # Disabled - too many false positives on documents
    "edge": 0.04,
    "local_variance": 0.04,
    "channel": 0.04,
    "jpeg_ghost": 0.05,
    "metadata": 0.05,
    "text_region": 0.10,
    "luminance": 0.05,
    # Document-specific (50% weight)
    "font_consistency": 0.12,
    "background_uniformity": 0.08,
    "text_alignment": 0.08,
    "micro_pattern": 0.07,
    "header_name": 0.15,  # HIGH weight - names are commonly altered
}

ANALYSIS_WEIGHTS = PHOTO_WEIGHTS  # Default


def run_full_analysis(img, image_path=None, force_document_mode=None):
    """
    Run all 10 forensic analyses on a single image.
    Returns a results dictionary with scores and visual outputs.
    """
    results = {}

    # 1. ELA
    ela_img, ela_mean, ela_hot = error_level_analysis(img)
    ela_score = min(100, ela_hot * 1000 + ela_mean * 2)
    # Multi-quality consistency boost
    mq = multi_quality_ela(img)
    consistent = sum(1 for _, _, hr in mq if hr > 0.01)
    if consistent >= 3:
        ela_score = min(100, ela_score + 15)
    results["ela"] = {
        "image": ela_img, "mean_error": round(ela_mean, 2),
        "hot_ratio": round(ela_hot, 4), "score": round(ela_score, 1),
        "multi_quality": mq,
    }

    # 2. Noise
    noise_map, noise_anom, noise_score = noise_analysis(img)
    results["noise"] = {
        "map": noise_map, "anomaly_map": noise_anom,
        "score": round(noise_score, 1),
    }

    # 3. Copy-Move
    cm_img, cm_matches, cm_score = copy_move_detection(img)
    results["copy_move"] = {
        "image": cm_img, "matches": cm_matches,
        "score": round(cm_score, 1),
    }

    # 4. Edge Anomaly
    edge_img, edge_anom, edge_score = edge_anomaly_detection(img)
    results["edge"] = {
        "image": edge_img, "anomaly_map": edge_anom,
        "score": round(edge_score, 1),
    }

    # 5. Local Variance
    var_map, var_anom, var_score = local_variance_analysis(img)
    results["local_variance"] = {
        "map": var_map, "anomaly_map": var_anom,
        "score": round(var_score, 1),
    }

    # 6. Channel Anomaly
    ch_map, ch_anom, ch_score = channel_anomaly_analysis(img)
    results["channel"] = {
        "map": ch_map, "anomaly_map": ch_anom,
        "score": round(ch_score, 1),
    }

    # 7. JPEG Ghost
    ghost_map, suspicious_q, ghost_score = jpeg_ghost_detection(img)
    results["jpeg_ghost"] = {
        "residual_curve": ghost_map, "suspicious_quality": suspicious_q,
        "score": round(ghost_score, 1),
    }

    # 8. Metadata
    if image_path:
        meta, meta_score = metadata_analysis(image_path)
    else:
        meta, meta_score = {}, 0.0
    results["metadata"] = {**meta, "score": round(meta_score, 1)}

    # 9. Text Region
    heatmap, regions, text_score = text_region_analysis(img)
    results["text_region"] = {
        "heatmap": heatmap,
        "regions": [
            {"x": r[0], "y": r[1], "w": r[2], "h": r[3], "score": round(r[4], 1)}
            for r in regions
        ],
        "score": round(text_score, 1),
    }

    # 10. Luminance Gradient
    grad_mag, lum_anom, lum_score = luminance_gradient_analysis(img)
    results["luminance"] = {
        "gradient": grad_mag, "anomaly_map": lum_anom,
        "score": round(lum_score, 1),
    }

    # --- Detect if this is a document or photo ---
    is_doc, doc_confidence = is_document_image(img)

    # Override detection if user forces a mode
    if force_document_mode is not None:
        is_doc = force_document_mode

    results["is_document"] = is_doc
    results["document_confidence"] = round(doc_confidence, 2)

    # --- Document-specific analyses (only run if document detected) ---
    if is_doc:
        # 11. Font Consistency
        font_anomalies, font_score, all_font_features = font_consistency_analysis(img)
        results["font_consistency"] = {
            "anomalies": [
                {"bbox": a["bbox"], "anomaly_score": round(a["anomaly_score"], 2)}
                for a in font_anomalies[:10]
            ],
            "score": round(font_score, 1),
        }

        # 12. Background Uniformity
        bg_anomalies, bg_score = background_uniformity_analysis(img)
        results["background_uniformity"] = {
            "anomalies": [
                {"bbox": a["bbox"], "z_score": round(a["z_score"], 2)}
                for a in bg_anomalies[:10]
            ],
            "score": round(bg_score, 1),
        }

        # 13. Text Alignment
        align_issues, align_score = text_alignment_analysis(img)
        results["text_alignment"] = {
            "misalignments": [
                {"bbox": a["bbox"], "deviation": round(a["baseline_deviation"], 1)}
                for a in align_issues[:10]
            ],
            "score": round(align_score, 1),
        }

        # 14. Micro Pattern Analysis
        micro_map, micro_anomalies, micro_score = micro_pattern_analysis(img)
        results["micro_pattern"] = {
            "anomaly_count": len(micro_anomalies),
            "score": round(micro_score, 1),
        }

        # 15. Header/Name Analysis (specifically for pay stubs, invoices)
        header_findings, header_score, header_features = header_name_analysis(img)
        results["header_name"] = {
            "findings": header_findings,
            "score": round(header_score, 1),
        }

        # Use document weights
        weights = DOCUMENT_WEIGHTS
    else:
        # Zero out document-specific scores for photos
        results["font_consistency"] = {"anomalies": [], "score": 0.0}
        results["background_uniformity"] = {"anomalies": [], "score": 0.0}
        results["text_alignment"] = {"misalignments": [], "score": 0.0}
        results["micro_pattern"] = {"anomaly_count": 0, "score": 0.0}
        results["header_name"] = {"findings": [], "score": 0.0}
        weights = PHOTO_WEIGHTS

    # --- Weighted overall score ---
    weighted = sum(
        results.get(k, {}).get("score", 0) * w
        for k, w in weights.items()
    )
    overall = min(100.0, weighted)

    # For documents, adjust thresholds (they need to be more sensitive)
    if is_doc:
        if overall >= 45:
            verdict = "HIGH - Very likely manipulated"
        elif overall >= 25:
            verdict = "MEDIUM - Possibly manipulated, review recommended"
        elif overall >= 10:
            verdict = "LOW - Minor anomalies detected"
        else:
            verdict = "CLEAN - No significant manipulation detected"
    else:
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
        "mode": "DOCUMENT" if is_doc else "PHOTO",
    }
    return results


# ============================================================
#  VISUALIZATION (Colab-friendly)
# ============================================================

def display_results(img, results, page_label=""):
    """Display rich visual results in Colab."""

    overall = results["overall"]
    score = overall["score"]
    verdict = overall["verdict"]
    mode = overall.get("mode", "PHOTO")
    is_doc = results.get("is_document", False)

    # Color for verdict - use document thresholds if document
    if is_doc:
        if score >= 45:
            color = "#e74c3c"
        elif score >= 25:
            color = "#f39c12"
        else:
            color = "#27ae60"
    else:
        if score >= 65:
            color = "#e74c3c"
        elif score >= 40:
            color = "#f39c12"
        else:
            color = "#27ae60"

    # Mode badge color
    mode_color = "#9b59b6" if is_doc else "#3498db"

    # Header
    display(HTML(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 20px; border-radius: 10px; margin: 10px 0;
                font-family: monospace; color: white;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h2 style="margin:0;">{'Fraud Detection Report' + (' - ' + page_label if page_label else '')}</h2>
            <span style="background: {mode_color}; padding: 5px 15px; border-radius: 20px;
                         font-size: 12px; font-weight: bold;">{mode} MODE</span>
        </div>
        <h1 style="color: {color}; margin: 10px 0;">
            Score: {score:.1f} / 100
        </h1>
        <h3 style="color: {color}; margin: 0;">{verdict}</h3>
    </div>
    """))

    # Get the right weights for display
    weights = DOCUMENT_WEIGHTS if is_doc else PHOTO_WEIGHTS

    # Score breakdown table - base analyses
    rows = ""
    labels = {
        "ela": "Error Level Analysis",
        "noise": "Noise Consistency",
        "copy_move": "Copy-Move Detection",
        "edge": "Edge Anomalies",
        "local_variance": "Local Variance",
        "channel": "Channel Anomalies",
        "jpeg_ghost": "JPEG Ghost Detection",
        "metadata": "Metadata Forensics",
        "text_region": "Text Region Anomalies",
        "luminance": "Luminance Gradient",
    }

    # Add document-specific labels
    if is_doc:
        labels["font_consistency"] = "Font Consistency"
        labels["background_uniformity"] = "Background Uniformity"
        labels["text_alignment"] = "Text Alignment"
        labels["micro_pattern"] = "Micro Pattern Analysis"
        labels["header_name"] = "Header/Name Analysis"

    for key, label in labels.items():
        s = results.get(key, {}).get("score", 0)
        w = weights.get(key, 0) * 100
        bar_width = int(s * 2)
        if s >= 50:
            bar_color = "#e74c3c"
        elif s >= 25:
            bar_color = "#f39c12"
        else:
            bar_color = "#27ae60"
        rows += f"""
        <tr>
            <td style="padding:4px 8px;">{label}</td>
            <td style="padding:4px 8px; text-align:center;">{w:.0f}%</td>
            <td style="padding:4px 8px;">
                <div style="background:#eee; border-radius:4px; width:200px; height:18px;">
                    <div style="background:{bar_color}; width:{bar_width}px;
                                height:18px; border-radius:4px;
                                text-align:center; color:white; font-size:11px;
                                line-height:18px;">{s:.1f}</div>
                </div>
            </td>
        </tr>
        """

    display(HTML(f"""
    <table style="border-collapse:collapse; font-family:monospace; font-size:13px;
                  margin: 10px 0;">
        <tr style="background:#1a1a2e; color:white;">
            <th style="padding:6px 8px;">Analysis</th>
            <th style="padding:6px 8px;">Weight</th>
            <th style="padding:6px 8px;">Score (0-100)</th>
        </tr>
        {rows}
    </table>
    """))

    # Visual outputs
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.25)

    # Original
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img)
    ax.set_title("Original Image", fontsize=11, fontweight="bold")
    ax.axis("off")

    # ELA
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(results["ela"]["image"])
    ax.set_title(f"ELA (score={results['ela']['score']})", fontsize=11, fontweight="bold")
    ax.axis("off")

    # Noise Map
    ax = fig.add_subplot(gs[0, 2])
    nm = results["noise"]["map"]
    ax.imshow(nm, cmap=FRAUD_CMAP, interpolation="nearest")
    ax.set_title(f"Noise Map (score={results['noise']['score']})", fontsize=11, fontweight="bold")
    ax.axis("off")

    # Copy-Move
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(results["copy_move"]["image"])
    ax.set_title(f"Copy-Move ({results['copy_move']['matches']} matches)", fontsize=11, fontweight="bold")
    ax.axis("off")

    # Edge
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(results["edge"]["image"], cmap="gray")
    ax.set_title(f"Edge Detection (score={results['edge']['score']})", fontsize=11, fontweight="bold")
    ax.axis("off")

    # Channel Anomaly
    ax = fig.add_subplot(gs[1, 2])
    cm = results["channel"]["map"]
    ax.imshow(cm, cmap=FRAUD_CMAP, interpolation="nearest")
    ax.set_title(f"Channel Anomaly (score={results['channel']['score']})", fontsize=11, fontweight="bold")
    ax.axis("off")

    # Text Region Heatmap
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(img, alpha=0.5)
    hm = results["text_region"]["heatmap"]
    ax.imshow(hm, cmap=FRAUD_CMAP, alpha=0.6)
    ax.set_title(f"Text Region Heatmap (score={results['text_region']['score']})",
                 fontsize=11, fontweight="bold")
    ax.axis("off")

    # Suspicious regions overlay
    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(img)
    for reg in results["text_region"]["regions"][:10]:
        rect = plt.Rectangle((reg["x"], reg["y"]), reg["w"], reg["h"],
                              linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
    ax.set_title(f"Suspicious Regions ({len(results['text_region']['regions'])} found)",
                 fontsize=11, fontweight="bold")
    ax.axis("off")

    # JPEG Ghost curve
    ax = fig.add_subplot(gs[2, 2])
    ghost = results["jpeg_ghost"]
    curve = ghost["residual_curve"]
    qs = sorted(curve.keys())
    vals = [curve[q] for q in qs]
    ax.plot(qs, vals, "b-o", markersize=4)
    if ghost["suspicious_quality"]:
        sq = ghost["suspicious_quality"]
        ax.axvline(x=sq, color="red", linestyle="--", label=f"Ghost @ Q={sq}")
        ax.legend()
    ax.set_xlabel("JPEG Quality")
    ax.set_ylabel("MSE Residual")
    ax.set_title(f"JPEG Ghost (score={ghost['score']})", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Forensic Analysis Visualizations", fontsize=14, fontweight="bold", y=1.01)
    plt.show()

    # Metadata findings
    meta = results.get("metadata", {})
    if meta.get("editing_software_detected"):
        display(HTML(f"""
        <div style="background:#e74c3c; color:white; padding:10px; border-radius:6px;
                    font-family:monospace; margin:10px 0;">
            WARNING: Editing software detected in metadata: <b>{meta.get('software','')}</b>
        </div>
        """))

    if meta.get("date_mismatch"):
        display(HTML("""
        <div style="background:#f39c12; color:white; padding:10px; border-radius:6px;
                    font-family:monospace; margin:10px 0;">
            WARNING: Date mismatch between creation and modification timestamps
        </div>
        """))

    # Suspicious text regions detail
    regions = results["text_region"]["regions"]
    if regions:
        display(HTML(f"""
        <div style="background:#16213e; color:white; padding:12px; border-radius:6px;
                    font-family:monospace; margin:10px 0;">
            <b>Top Suspicious Text Regions (potential altered text/amounts):</b><br>
            {'<br>'.join(
                f'  Region #{i+1}: position=({r["x"]},{r["y"]}) '
                f'size={r["w"]}x{r["h"]} anomaly_score={r["score"]}'
                for i, r in enumerate(regions[:10])
            )}
        </div>
        """))

    # --- Document-specific findings ---
    if is_doc:
        doc_findings = []

        # Header/Name analysis (most important for pay stubs)
        header_findings = results.get("header_name", {}).get("findings", [])
        header_score = results.get("header_name", {}).get("score", 0)
        if header_findings or header_score > 20:
            doc_findings.append(f"<b>HEADER/NAME AREA:</b> Score {header_score:.0f}/100")
            for hf in header_findings:
                doc_findings.append(f"  - {hf}")

        # Font consistency issues
        font_anomalies = results.get("font_consistency", {}).get("anomalies", [])
        if font_anomalies:
            doc_findings.append(f"<b>Font Inconsistencies:</b> {len(font_anomalies)} region(s) with different text rendering")

        # Background uniformity issues
        bg_anomalies = results.get("background_uniformity", {}).get("anomalies", [])
        if bg_anomalies:
            doc_findings.append(f"<b>Background Issues:</b> {len(bg_anomalies)} region(s) with inconsistent background")

        # Text alignment issues
        align_issues = results.get("text_alignment", {}).get("misalignments", [])
        if align_issues:
            doc_findings.append(f"<b>Alignment Issues:</b> {len(align_issues)} text region(s) misaligned from baseline")

        # Micro pattern anomalies
        micro_count = results.get("micro_pattern", {}).get("anomaly_count", 0)
        if micro_count > 5:
            doc_findings.append(f"<b>Micro Pattern Anomalies:</b> {micro_count} suspicious micro-level pattern breaks")

        if doc_findings:
            display(HTML(f"""
            <div style="background:#9b59b6; color:white; padding:12px; border-radius:6px;
                        font-family:monospace; margin:10px 0;">
                <b style="font-size:14px;">DOCUMENT-SPECIFIC FINDINGS:</b><br><br>
                {'<br>'.join(doc_findings)}
            </div>
            """))

        # Show font anomaly details
        if font_anomalies:
            display(HTML(f"""
            <div style="background:#2c3e50; color:white; padding:12px; border-radius:6px;
                        font-family:monospace; margin:10px 0;">
                <b>Font Rendering Anomalies (text may have been replaced):</b><br>
                {'<br>'.join(
                    f'  Text region at ({a["bbox"][0]},{a["bbox"][1]}) '
                    f'size={a["bbox"][2]}x{a["bbox"][3]} - anomaly_score={a["anomaly_score"]}'
                    for a in font_anomalies[:10]
                )}
            </div>
            """))


# ============================================================
#  MAIN ENTRY POINT
# ============================================================

def analyze_file(file_path):
    """Analyze an uploaded image or PDF file."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        from pdf2image import convert_from_path
        pages = convert_from_path(file_path, 300)
        print(f"PDF loaded: {len(pages)} page(s)")
        for i, page in enumerate(pages):
            print(f"\n{'='*60}")
            print(f"  Processing page {i+1} of {len(pages)}")
            print(f"{'='*60}")
            img = page.convert("RGB")
            results = run_full_analysis(img)
            display_results(img, results, page_label=f"Page {i+1}")
    else:
        img = Image.open(file_path).convert("RGB")
        results = run_full_analysis(img, image_path=file_path)
        display_results(img, results)


# =============================================
# CELL 3: RUN - Upload and analyze
# =============================================
# Paste this into the third Colab cell:
"""
from google.colab import files

print("Upload an image (JPG/PNG) or PDF to analyze for fraud:")
uploaded = files.upload()

for filename in uploaded.keys():
    print(f"\nAnalyzing: {filename}")
    analyze_file(filename)
"""

# If running as a script (not Colab), support command-line usage
def _is_notebook():
    try:
        get_ipython()
        return True
    except NameError:
        return False

if __name__ == "__main__" and not _is_notebook():
    import sys
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            analyze_file(f)
    else:
        print("Usage: python fraud_detection_colab.py <image_or_pdf>")
        print("Or copy the code into Google Colab (see instructions at top of file).")
