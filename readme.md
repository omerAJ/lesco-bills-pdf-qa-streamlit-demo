# ⚡ Electricity Bill Q&A (Streamlit)

Tiny app that:

- Lets a user upload a bill PDF
- Auto-crops a meter ROI and asks: “What is the reading on the meter?”
- Then switches to the full PDF for the rest of the chat

Built with **Streamlit**, **OpenAI Responses API**, **PyMuPDF**, and **Pillow**.

---

## Features

- **One-click start:** upload a PDF, get the meter reading.
- **ROI first** for reliable reading → then full-document chat.
- **Grounded answers** (model only uses the uploaded file).

---

## Project Layout

```
.
├─ app.py               # Streamlit UI & flow
├─ agent_factory.py     # Responses API + system prompt
├─ utils.py             # crop_roi_to_pdf()
├─ requirements.txt
└─ README.md
```

---

## Quickstart

### Prereqs

- Python 3.10+
- OpenAI API key (`OPENAI_API_KEY`)
- Packages: `streamlit`, `openai>=1.40.0`, `langchain`, `PyMuPDF`, `Pillow`

### Install & run

```bash
python -m venv .venv
# Activate the virtual environment:
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...       # Windows (Powershell): $env:OPENAI_API_KEY="sk-..."
streamlit run app.py
```

---

## How it Works

1. **Upload a PDF bill.**
2. The app crops a predefined ROI (meter window) and asks the model for the reading.
3. After the first reply, it replaces the ROI file with the full PDF for continued Q&A.

---

## Configuration

**ROI (adjust to your bill template) — in `app.py`:**

```python
ROI_BBOX = (348, 469, 540, 610)  # (x0, y0, x1, y1) PDF points, origin bottom-left
```

**Optional:** increase DPI for low-res scans:

```python
crop_roi_to_pdf(..., dpi=500)
```

---

## Streamlit Cloud Notes

- Add `OPENAI_API_KEY` in Secrets.
- Install PyMuPDF (imports as `fitz`).

---

## Troubleshooting

- **Module “fitz” not found:** install PyMuPDF, not fitz.
- **400 invalid input type:** Responses API requires input_text/input_file (already used here).
- **Blurry reads:** raise dpi in `crop_roi_to_pdf()` or lightly sharpen in `utils.py`.

---

## License

MIT. PRs welcome.