import pandas as pd
import docx
import PyPDF2


def read_docx(file):
    doc = docx.Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)


def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    full_text = []
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        full_text.append(page.extract_text())
    return '\n'.join(full_text)


def read_file(file):
    if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":  # .docx
        text = read_docx(file)
    elif file.type == "application/pdf":  # .pdf
        text = read_pdf(file)
    else:
        return None

    return text


def get_similar_docs(index_pc, query: str, k: int = 2):
    similar_docs = index_pc.similarity_search(query, k=2)
    return similar_docs

