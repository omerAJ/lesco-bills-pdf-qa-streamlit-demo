# preview_roi_pdf.py
import argparse
import os
import sys
import subprocess
from utils import crop_roi_to_pdf

def open_file(path: str):
    try:
        if sys.platform.startswith("darwin"):
            subprocess.run(["open", path], check=False)
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as e:
        print(f"Couldn't auto-open the file: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Crop a PDF ROI and open the resulting one-page PDF."
    )
    parser.add_argument("--pdf", help="Input PDF path")
    parser.add_argument(
        "--out",
        help="Output PDF path (default: roi_<inputname>.pdf)",
        default=None,
    )
    parser.add_argument(
        "--page", type=int, default=0, help="Zero-based page index (default: 0)"
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=(348, 469, 540, 610),
        help="Bounding box in PDF points, origin BOTTOM-LEFT (x0 y0 x1 y1)",
    )
    parser.add_argument("--dpi", type=int, default=400, help="Render DPI (default: 400)")
    parser.add_argument(
        "--no-sharpen",
        action="store_true",
        help="Disable the unsharp mask cleanup",
    )
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        raise FileNotFoundError(args.pdf)

    out_path = args.out or f"roi_{os.path.basename(args.pdf)}"
    bbox = tuple(args.bbox) if args.bbox else (100, 500, 200, 550)

    result = crop_roi_to_pdf(
        in_pdf_path=args.pdf,
        out_pdf_path=out_path,
        page_number=args.page,
        bbox=bbox,
        dpi=args.dpi,
        sharpen=not args.no_sharpen,
    )

    print(f"Saved ROI PDF -> {result}")
    open_file(result)

if __name__ == "__main__":
    main()
