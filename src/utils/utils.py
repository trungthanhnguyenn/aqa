from pypdf import PdfReader
import docx

def read_file(file_path):
    if file_path.endswith(".pdf"):
        pdf = PdfReader(file_path)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text
        
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    
    elif file_path.endswith(".txt"):
        # Read text file
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file format. Only .txt files are supported.")