from enum import Enum
import PyPDF2

class PAGE(Enum):
    MAIN = 1
    SUMMARY = 2
    MCQ = 3
    CHAT = 4


# Function to read PDF content
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text
