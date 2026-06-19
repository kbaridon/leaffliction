import cv2
import numpy as np
import matplotlib.pyplot as plt


MIN_CORNER_SIZE = 8
A_CHANNEL_FLOOR = 125
L_STRICT_DROP = 25.0
A_RELAX_MARGIN = 6
L_RELAX_DROP = 12
MIN_LESION_AREA = 30

N_PSEUDOLANDMARKS = 20
SATURATION = 60
HIST_BINS = 256


def _get_leaf_mask(img):
    """Return a cleaned binary mask of the whole leaf (foreground)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = img.shape[:2]

    corner_size = max(MIN_CORNER_SIZE, min(h, w) // 20)
    corners = np.concatenate([
        lab[:corner_size, :corner_size].reshape(-1, 3),
        lab[:corner_size, -corner_size:].reshape(-1, 3),
        lab[-corner_size:, :corner_size].reshape(-1, 3),
        lab[-corner_size:, -corner_size:].reshape(-1, 3),
    ])
    bg = np.median(corners, axis=0)

    diff = np.sqrt(
        (lab[..., 1] - bg[1]) ** 2 + (lab[..., 2] - bg[2]) ** 2
    )
    diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(
        diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if n_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if contours:
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        mask = filled

    return mask


def _get_disease_mask(img):
    """Return a mask of diseased (non-green) regions inside the leaf."""
    leaf = _get_leaf_mask(img)
    if not np.any(leaf):
        return leaf

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    leaf_inner = cv2.erode(leaf, erode_kernel, iterations=1)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    a = lab[..., 1]
    inner_a = a[leaf_inner > 0]
    inner_L = L[leaf_inner > 0]
    if inner_a.size == 0:
        return np.zeros_like(leaf)

    otsu_a, _ = cv2.threshold(
        inner_a.reshape(1, -1).astype(np.uint8),
        0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    a_thr = max(int(otsu_a), A_CHANNEL_FLOOR)
    L_med = float(np.median(inner_L))
    L_thr = L_med - L_STRICT_DROP

    seeds = ((a > a_thr) | (L < L_thr)) & (leaf_inner > 0)
    seeds = seeds.astype(np.uint8) * 255

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, open_kernel, iterations=1)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        seeds, connectivity=8
    )
    keep = np.where(stats[1:, cv2.CC_STAT_AREA] >= MIN_LESION_AREA)[0] + 1
    seeds = np.isin(labels, keep).astype(np.uint8) * 255

    relaxed = ((a > a_thr - A_RELAX_MARGIN) |
               (L < L_med - L_RELAX_DROP)) & (leaf_inner > 0)
    relaxed = relaxed.astype(np.uint8) * 255

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    seed_zone = cv2.dilate(seeds, dilate_kernel, iterations=1)
    disease = cv2.bitwise_and(relaxed, seed_zone)
    return cv2.bitwise_or(disease, seeds)


def get_gaussian_blur(img):
    """Take an image and return a gaussian blur transformation of it."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1]
    _, s_thresh = cv2.threshold(s, SATURATION, 255, cv2.THRESH_BINARY)
    return cv2.GaussianBlur(s_thresh, (5, 5), 0)


def get_mask(img):
    """Take an image and return a mask transformation of it."""
    mask = _get_disease_mask(img)
    masked = img.copy()
    masked[mask == 0] = 255
    return cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)


def get_roi(img):
    """Take an image and return the roi of it."""
    mask = _get_disease_mask(img)
    h, w = img.shape[:2]
    output = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output[mask > 0] = (0, 255, 0)
    cv2.rectangle(output, (0, 0), (w - 1, h - 1), (255, 0, 0), 5)
    return output


def get_analyze(img):
    """Take an image and return a analyze objects transformation of it."""
    mask = _get_leaf_mask(img)
    output = img.copy()
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
    if contours:
        main = max(contours, key=cv2.contourArea)
        cv2.drawContours(output, [main], -1, (255, 0, 255), 2)
        moments = cv2.moments(main)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


def _strip_landmarks(mask, n):
    """For each vertical strip, return (top, bottom, centroid) points."""
    _, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [], [], []
    x_min, x_max = int(xs.min()), int(xs.max())
    step = (x_max - x_min + 1) / n
    tops, bots, ctrs = [], [], []
    for i in range(n):
        x_start = int(x_min + i * step)
        x_end = max(int(x_min + (i + 1) * step), x_start + 1)
        strip_ys, strip_xs = np.where(mask[:, x_start:x_end] > 0)
        if len(strip_ys) == 0:
            continue
        strip_xs = strip_xs + x_start
        top_idx = int(np.argmin(strip_ys))
        bot_idx = int(np.argmax(strip_ys))
        tops.append((int(strip_xs[top_idx]), int(strip_ys[top_idx])))
        bots.append((int(strip_xs[bot_idx]), int(strip_ys[bot_idx])))
        ctrs.append((int(np.mean(strip_xs)), int(np.mean(strip_ys))))
    return tops, bots, ctrs


def get_pseudolandmarks(img):
    """Take an image and return a pseudolandmarks transformation of it."""
    mask = _get_leaf_mask(img)
    tops, bots, v_ctrs = _strip_landmarks(mask, N_PSEUDOLANDMARKS)
    lefts_t, rights_t, h_ctrs_t = _strip_landmarks(mask.T, N_PSEUDOLANDMARKS)
    lefts = [(y, x) for x, y in lefts_t]
    rights = [(y, x) for x, y in rights_t]
    h_ctrs = [(y, x) for x, y in h_ctrs_t]

    output = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    point_sets = [
        (tops, (255, 0, 0)),
        (bots, (255, 0, 255)),
        (v_ctrs, (0, 79, 255)),
        (lefts, (0, 163, 255)),
        (rights, (0, 255, 0)),
        (h_ctrs, (255, 255, 0)),
    ]
    for pts, color in point_sets:
        for x, y in pts:
            cv2.circle(output, (x, y), 3, color, -1)
    return output


_HIST_CHANNELS = [
    ("blue", "blue"),
    ("blue-yellow", "yellow"),
    ("green", "green"),
    ("green-magenta", "magenta"),
    ("hue", "purple"),
    ("lightness", "gray"),
    ("red", "red"),
    ("saturation", "cyan"),
    ("value", "orange"),
]


def _extract_channels(img):
    """Return per-channel arrays for RGB, HSV and LAB color spaces."""
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return {
        "blue": b,
        "blue-yellow": lab[..., 2],
        "green": g,
        "green-magenta": lab[..., 1],
        "hue": hsv[..., 0],
        "lightness": lab[..., 0],
        "red": r,
        "saturation": hsv[..., 1],
        "value": hsv[..., 2],
    }


def get_color_histogram(img):
    """Take an image and return a color histogram transformation of it."""
    mask = _get_leaf_mask(img)
    selected = mask > 0
    channels = _extract_channels(img)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, color in _HIST_CHANNELS:
        data = channels[label][selected]
        if data.size == 0:
            continue
        hist, bins = np.histogram(data, bins=HIST_BINS, range=(0, HIST_BINS))
        proportion = hist / hist.sum() * 100
        ax.plot(bins[:-1], proportion, label=label, color=color)

    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Proportion of pixels (%)")
    ax.set_xlim(0, 255)
    ax.legend(loc="best", fontsize="x-small")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return arr
