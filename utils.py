# utils_pdf.py
import fitz  # PyMuPDF
from PIL import Image, ImageFilter, ImageOps
import io

def crop_roi_to_pdf(in_pdf_path, out_pdf_path, page_number=0, bbox=(100, 500, 200, 550), dpi=400, sharpen=True):
    """
    bbox uses PDF coordinates in points (1/72 inch), origin at bottom-left:
      (x0, y0, x1, y1)
    """
    doc = fitz.open(in_pdf_path)
    page = doc[page_number]
    rect = fitz.Rect(*bbox)

    # Render the ROI at high DPI to get a crisp bitmap
    pix = page.get_pixmap(clip=rect, dpi=dpi)

    # Optional: light cleanup for low-res scans
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    if sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=160))

    # Encode as PNG bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    # Create a new 1-page PDF sized to the image and place the PNG
    out = fitz.open()
    page_w, page_h = img.width, img.height  # pixels become points (1:1 here)
    new_page = out.new_page(width=page_w, height=page_h)
    new_page.insert_image(fitz.Rect(0, 0, page_w, page_h), stream=png_bytes)
    out.save(out_pdf_path)
    out.close()
    doc.close()
    return out_pdf_path
