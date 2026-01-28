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
#  COMPOSITE FRAUD DETECTOR
# ============================================================

ANALYSIS_WEIGHTS = {
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


def run_full_analysis(img, image_path=None):
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

    # Weighted overall
    weighted = sum(
        results.get(k, {}).get("score", 0) * w
        for k, w in ANALYSIS_WEIGHTS.items()
    )
    overall = min(100.0, weighted)

    if overall >= 65:
        verdict = "HIGH - Very likely manipulated"
    elif overall >= 40:
        verdict = "MEDIUM - Possibly manipulated, review recommended"
    elif overall >= 20:
        verdict = "LOW - Minor anomalies detected"
    else:
        verdict = "CLEAN - No significant manipulation detected"

    results["overall"] = {"score": round(overall, 1), "verdict": verdict}
    return results


# ============================================================
#  VISUALIZATION (Colab-friendly)
# ============================================================

def display_results(img, results, page_label=""):
    """Display rich visual results in Colab."""

    overall = results["overall"]
    score = overall["score"]
    verdict = overall["verdict"]

    # Color for verdict
    if score >= 65:
        color = "#e74c3c"
    elif score >= 40:
        color = "#f39c12"
    else:
        color = "#27ae60"

    # Header
    display(HTML(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 20px; border-radius: 10px; margin: 10px 0;
                font-family: monospace; color: white;">
        <h2 style="margin:0;">{'Fraud Detection Report' + (' - ' + page_label if page_label else '')}</h2>
        <h1 style="color: {color}; margin: 10px 0;">
            Score: {score:.1f} / 100
        </h1>
        <h3 style="color: {color}; margin: 0;">{verdict}</h3>
    </div>
    """))

    # Score breakdown table
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
    for key, label in labels.items():
        s = results.get(key, {}).get("score", 0)
        w = ANALYSIS_WEIGHTS.get(key, 0) * 100
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
