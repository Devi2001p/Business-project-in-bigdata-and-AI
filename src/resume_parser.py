"""
This file is used in handling the text from the resume, basically, to read, extract and clean the text
The resumes are taken in form of pdf, docx or txt.

"""
from typing import Optional
import pathlib
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract
import docx
from .preprocess import text_that_is_cleaned

Formats_supported = {".pdf", ".docx", ".txt"}

def to_read_the_pdf(path: str) -> str:
    """
    This function is used to extract from the pdf files with help of PyPDF2 (The fallback is pdfminer)
    """
    try:
        the_reader = PdfReader(path)
        textual_parts = []
        for page in the_reader.pages:
            textual_parts.append(page.extract_text() or "")
        text = "\n".join(textual_parts)
        if text.strip():
            return text_that_is_cleaned(text)
    except Exception:
        pass

    # fallback to pdfminer
    try:
        text = pdfminer_extract(path)
        return text_that_is_cleaned(text or "")
    except Exception as e:
        print(f"❌ PDF extraction failed for {path}: {e}")
        return ""


def to_read_the_docx(path: str) -> str:
    """
    This function is used to extract the text from docx file with help of python-docx.
    """
    try:
        d = docx.Document(path)
        text = "\n".join([p.text for p in d.paragraphs])
        return text_that_is_cleaned(text)
    except Exception as e:
        print(f"❌ Extraction of the given docx file is failed: {path}: {e}")
        return ""


def to_read_the_txt(path: str) -> str:
    """
    This function is used to extract text from the plain txt files.
    """
    try:
        return text_that_is_cleaned(pathlib.Path(path).read_text(encoding="utf-8", errors="ignore"))
    except Exception as e:
        print(f"❌ Extraction of the given txt file is failed: {path}: {e}")
        return ""


def to_parse_the_resume(path: str) -> str:
    """
    This function is used to parse the text in the resume by depending on the extension of the files
    """
    ext = pathlib.Path(path).suffix.lower()
    if ext == ".pdf":
        return to_read_the_pdf(path)
    elif ext == ".docx":
        return to_read_the_docx(path)
    elif ext == ".txt":
        return to_read_the_txt(path)
    else:
        raise ValueError(
            f"❌ Given file type is not supported: {ext}. formats that are supported: {', '.join(Formats_supported)}"
        )