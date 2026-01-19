import re 
import os 
from dotenv import load_dotenv

load_dotenv()

SECTION_PATTERNS = {
    "case_details": r"(case\s+details)",
    "issues": r"(issues?\s+(for\s+consideration|involved))",
    "headnotes": r"(head\s*notes?)",
    "citations": r"(citations?|case law cited)",
    "acts" : r"(list of acts?)",
    "keywords": r"(list of keywords?|keywords)",
    "arising" : r"((case\s+arising\s+out\s+of)|(case\s+arising\s+from))",
    "appearances": r"(appearances\s+for\s+parties)",
    "judgment": r"(order of the supreme court|judgement)"
}

SECTION_ORDER = ["case_details", "issues", "headnotes", "citations", "acts", "keywords", "other_details", "judgment"]

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"Digital Supreme Court Reports", "", text)
    return text.strip()

def split_into_sections(text: str):
    text = normalize_text(text)

    matches = []
    for section, pattern in SECTION_PATTERNS.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append({
                "section": section,
                "start": match.start()
            })

    if not matches:
        return [{"section": "unknown", "text": text}]

    matches = sorted(matches, key=lambda x: x["start"])

    sections = []
    for i, current in enumerate(matches):
        start = current["start"]
        end = matches[i + 1]["start"] if i + 1 < len(matches) else len(text)

        section_text = text[start:end].strip()
        sections.append({
            "section": current["section"],
            "text": section_text
        })

    return sections


if(__name__ == "__main__"):
    
    NEW_DATA_2025_TEXT_FOLDER = os.getenv("NEW_DATA_2025_TEXT_FOLDER")
    sample_text_file_name = "ESCR010000032025.txt"

    print(NEW_DATA_2025_TEXT_FOLDER, sample_text_file_name)

    sample_text = ""
    with open(os.path.join(NEW_DATA_2025_TEXT_FOLDER, sample_text_file_name), "r", encoding="utf-8") as text_file:
        sample_text = text_file.read()

    sections = split_into_sections(sample_text)
    for sec in sections:
        print(f"Section: {sec['section']}\nText: {sec['text']}\n{'-'*40}\n")