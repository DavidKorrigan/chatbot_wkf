import fitz  # PyMuPDF
import sys

"""
This script produces a raw text file with all content from the PDF file pass in parameter.
Text file should be then clean manually or programmatically (e.g., remove headers/footers, fix spacing).
"""

def extract_pdf_text(pdf_path: str):
    """
    Extract all text from PDF file to text.
    """
    doc = fitz.open(pdf_path)
    full_text = []

    for page in doc:
        text = page.get_text()
        if text:
            full_text.append(text)

    return "\n".join(full_text)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract.py <input_pdf> <output_txt>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    text = extract_pdf_text(input_file)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    print("✔️ Extracted text saved to kumite_rules.txt")