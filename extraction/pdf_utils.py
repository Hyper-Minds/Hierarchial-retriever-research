import os
import pymupdf


def extractPDF(pdf_path : str) -> str:
    try:
        doc = pymupdf.open(pdf_path)
        pdf_text = ''
        for page_num in range(len(doc)):    
            page = doc.load_page(page_num)
            text = page.get_text()
            pdf_text = pdf_text + text
        return pdf_text
    except:
        print("Cannot process the file ",pdf_path )
    return ''

