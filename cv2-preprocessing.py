# preprocess_meter.py
# Goal: enhance seven-seg digits and preserve the tiny decimal point.
# Usage:
#   python preprocess_meter.py --img path/to/img.png --save out.png

import argparse, os
import cv2
import numpy as np

def preprocess(img_bgr, show_steps=False):
    # 2) grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 3) illumination normalization (blur-divide)
    # big kernel to estimate background
    bg = cv2.GaussianBlur(gray, (0,0), sigmaX=25, sigmaY=25)
    bg = np.clip(bg, 1, None)
    norm = cv2.divide(gray, bg, scale=255)

    # 4) contrast boost (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    hi = clahe.apply(norm)

    # 5) light denoise that won’t kill the dot
    # (avoid median with large kernel; it erases tiny points)
    den = cv2.bilateralFilter(hi, d=7, sigmaColor=25, sigmaSpace=7)

    # 6) DIGITS path: adaptive threshold
    digits = cv2.adaptiveThreshold(
        den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 7
    )
    # gentle open to break dust specks, keep segments
    digits = cv2.morphologyEx(digits, cv2.MORPH_OPEN, np.ones((2,2), np.uint8), iterations=1)

    # 7) DECIMAL path: black-hat to enhance tiny dark blobs on light background
    # use a tiny kernel so only dot-size survives
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    blackhat = cv2.morphologyEx(den, cv2.MORPH_BLACKHAT, k)
    # normalize then threshold small responses
    bh_norm = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    _, dot_mask = cv2.threshold(bh_norm, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # keep only very small components (areas typical of dot)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dot_mask, connectivity=8)
    cleaned_dot = np.zeros_like(dot_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 2 <= area <= 120:  # tune for your DPI; small spots only
            cleaned_dot[labels == i] = 255

    # 8) Combine
    combo = cv2.bitwise_or(digits, cleaned_dot)

    # OPTIONAL: thicken a touch to help OCR
    combo = cv2.morphologyEx(combo, cv2.MORPH_DILATE, np.ones((2,2), np.uint8), iterations=1)

    if show_steps:
        import matplotlib.pyplot as plt
        imgs = [img_bgr, gray, norm, hi, den, digits, bh_norm, cleaned_dot, combo]
        titles = ["original", "Gray", "Blur-Divide", "CLAHE", "Denoised",
                  "Digits (adaptive)", "Black-hat (norm)", "Dot mask", "Final mask"]
        for t, im in zip(titles, imgs):
            plt.figure(); 
            if im.ndim == 2:
                plt.imshow(im, cmap='gray')
            else:
                plt.imshow(im)
            plt.title(t); plt.axis('off')
        plt.show()

    return combo, roi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="Path to meter image")
    ap.add_argument("--save", default="preprocessed.png", help="Output mask path")
    ap.add_argument("--show", action="store_true", help="Show steps with matplotlib")
    args = ap.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(args.img)

    mask, roi = preprocess(img, show_steps=args.show)

    cv2.imwrite(args.save, mask)
    print(f"Saved -> {args.save}")

    # For quick visual check side-by-side
    vis = np.hstack([cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), mask])
    cv2.imwrite(os.path.splitext(args.save)[0] + "_sideby.png", vis)

if __name__ == "__main__":
    main()
