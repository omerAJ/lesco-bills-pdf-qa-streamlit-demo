# test-cropping.py
# Interactive PDF ROI picker to figure out bbox coordinates.
# - Click and drag on the page to select a region.
# - The cropped region pops up in a separate window.
# - Console prints the PDF bbox in points (origin TOP-LEFT).
# - Press 's' to save the crop (roi_preview.png) and bbox (bbox.txt).
# - Press 'q' to quit.
#
# Why this fixes the "crop is above selection" bug:
# - We use origin='upper' and extent=(0, W, H, 0) in imshow so data coords match image pixels exactly.
# - We DO NOT flip y. PyMuPDF's rendered pixmap and Matplotlib view both use top-left origin, y down.
# - We use the same scale matrix for the full render and the crop preview to avoid discrepancies.

import argparse
import os
import fitz  # PyMuPDF
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


def pixmap_to_ndarray(pix):
    """Convert a PyMuPDF Pixmap to a HxWx3 uint8 numpy array (drop alpha)."""
    if pix.alpha:
        pix = fitz.Pixmap(fitz.csRGB, pix)  # drop alpha channel
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return arr


def render_page(doc, page_number, dpi):
    """Render a PDF page to an image using a pure scale (no rotation)."""
    page = doc[page_number]
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)  # IMPORTANT: use same matrix for full render & crop
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = pixmap_to_ndarray(pix)
    return page, pix, img, scale, mat


def device_to_pdf_points(x_px, y_px, scale):
    """
    Convert device pixels (top-left origin, y down) to PDF points.
    Since we render with a pure scale, mapping is simply divide by scale.
    """
    x_pt = x_px / scale
    y_pt = y_px / scale
    return x_pt, y_pt


def main():
    parser = argparse.ArgumentParser(description="Interactively crop a region from a PDF to figure out bbox.")
    parser.add_argument("--pdf", help="Path to the PDF file")
    parser.add_argument("--page", type=int, default=0, help="Zero-based page index (default: 0)")
    parser.add_argument("--dpi", type=int, default=400, help="Render DPI (default: 400)")
    parser.add_argument("--bbox", nargs=4, type=float,
                        help="Optional bbox to preview directly: x0 y0 x1 y1 (PDF points, origin TOP-LEFT)")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        raise FileNotFoundError(args.pdf)

    doc = fitz.open(args.pdf)
    if args.page < 0 or args.page >= len(doc):
        raise ValueError(f"Page {args.page} out of range (document has {len(doc)} pages).")

    page, pix, img, scale, mat = render_page(doc, args.page, args.dpi)
    W, H = pix.width, pix.height
    page_w_pt = page.rect.width
    page_h_pt = page.rect.height

    print(f"Loaded: {args.pdf}")
    print(f"Page {args.page} size: {page_w_pt:.2f} x {page_h_pt:.2f} points; rendered {W} x {H} px at {args.dpi} DPI")
    if page.rotation:
        print(f"Note: Page has rotation flag {page.rotation}°, but we render without applying it for accurate mapping.")
        print("      The page may appear sideways; selection mapping will still be correct.")

    # State to hold the last crop for saving
    state = {"last_bbox_pts": None, "last_crop_img": None}

    def show_crop_preview(bbox_pts):
        crop_pix = page.get_pixmap(clip=fitz.Rect(*bbox_pts), matrix=mat, alpha=False)
        crop_img = pixmap_to_ndarray(crop_pix)
        plt.figure("Cropped Region Preview")
        plt.imshow(crop_img, origin="upper", interpolation="nearest")
        plt.axis("off")
        x0, y0, x1, y1 = bbox_pts
        plt.title(f"Crop @ {args.dpi} DPI\nPDF bbox (pts, origin TOP-LEFT): ({x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f})")
        plt.tight_layout()
        plt.show(block=False)
        state["last_bbox_pts"] = bbox_pts
        state["last_crop_img"] = crop_img

    def onselect(eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            print("Selection outside image axes; try again.")
            return

        # Selection in display pixels (top-left origin, y down)
        x0_px, y0_px = eclick.xdata, eclick.ydata
        x1_px, y1_px = erelease.xdata, erelease.ydata

        # Normalize pixel coords to min/max box
        x_min_px, x_max_px = sorted([x0_px, x1_px])
        y_min_px, y_max_px = sorted([y0_px, y1_px])

        # Clamp to image bounds (defensive)
        x_min_px = max(0, min(W, x_min_px))
        x_max_px = max(0, min(W, x_max_px))
        y_min_px = max(0, min(H, y_min_px))
        y_max_px = max(0, min(H, y_max_px))

        # Convert to PDF points (divide by scale, NO y-flip)
        x0_pt, y0_pt = device_to_pdf_points(x_min_px, y_min_px, scale)
        x1_pt, y1_pt = device_to_pdf_points(x_max_px, y_max_px, scale)

        bbox_pts = (x0_pt, y0_pt, x1_pt, y1_pt)

        # Print coords
        print("\n=== Selection ===")
        print(f"PDF bbox (points, origin TOP-LEFT): ({x0_pt:.2f}, {y0_pt:.2f}, {x1_pt:.2f}, {y1_pt:.2f})")
        print(f"As integers: ({int(round(x0_pt))}, {int(round(y0_pt))}, {int(round(x1_pt))}, {int(round(y1_pt))})")
        print("Tip: Add a few points of padding so digits aren’t clipped.")

        # Show preview rendered from PDF with the SAME matrix
        show_crop_preview(bbox_pts)

    def onkeypress(event):
        if event.key == 'q':
            plt.close('all')
        elif event.key == 's':
            if state["last_crop_img"] is None or state["last_bbox_pts"] is None:
                print("No selection to save yet.")
                return
            # Save crop image
            out_img = "roi_preview.png"
            plt.imsave(out_img, state["last_crop_img"])
            # Save bbox
            x0, y0, x1, y1 = state["last_bbox_pts"]
            out_txt = "bbox.txt"
            with open(out_txt, "w") as f:
                f.write(f"{x0:.2f} {y0:.2f} {x1:.2f} {y1:.2f}\n")
            print(f"Saved crop -> {out_img}")
            print(f"Saved bbox -> {out_txt}")

    # Optional: preview a known bbox
    if args.bbox:
        x0, y0, x1, y1 = map(float, args.bbox)
        show_crop_preview((x0, y0, x1, y1))

    # Show main page and enable rectangle selection.
    # extent=(0, W, H, 0) + origin='upper' aligns data coords to pixel edges (no half-pixel offset).
    fig, ax = plt.subplots(num="PDF Page Viewer (drag to select; 's' to save, 'q' to quit)")
    ax.imshow(img, extent=(0, W, H, 0), origin='upper', interpolation='nearest')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_axis_off()

    rs = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],          # left mouse button
        minspanx=3, minspany=3,
        spancoords='pixels',
        interactive=True
    )

    fig.canvas.mpl_connect('key_press_event', onkeypress)
    plt.tight_layout()
    print("Instructions:\n - Drag a rectangle on the page to select ROI\n - Press 's' to save crop & bbox\n - Press 'q' to quit")
    plt.show()


if __name__ == "__main__":
    main()
