from pypdf import PdfReader

# =========================
# LOAD PDF FILE
# =========================
def load_pdf(file_path):
    text = ""
    reader = PdfReader(file_path)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


# =========================
# LOAD TXT FILE
# =========================
def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()