"""
ingestion/pdf_extractor.py

Extracts clean structured text from Supreme Court and High Court
judgment PDFs. Handles multi-column layouts, header/footer removal,
and basic section detection.

Usage:
    python ingestion/pdf_extractor.py --input data/raw/judgment.pdf --output data/processed/
    python ingestion/pdf_extractor.py --input data/raw/ --output data/processed/  # batch
"""

import argparse
import json
import re
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import fitz  # pymupdf
from tqdm import tqdm
from rich.console import Console

console = Console()


@dataclass
class ExtractedJudgment:
    """Raw extracted text from a judgment PDF, before IRAC parsing."""
    filename: str
    full_text: str
    pages: int
    # Detected metadata (best-effort from header parsing)
    case_name: Optional[str] = None
    citation: Optional[str] = None
    court: Optional[str] = None
    year: Optional[int] = None
    date_of_judgment: Optional[str] = None
    coram: Optional[str] = None       # bench composition
    # Detected sections (best-effort)
    facts_text: Optional[str] = None
    issues_text: Optional[str] = None
    held_text: Optional[str] = None


# Patterns for Indian legal citations
CITATION_PATTERNS = [
    r'\(\d{4}\)\s+\d+\s+SCC\s+\d+',           # (2012) 9 SCC 1
    r'AIR\s+\d{4}\s+SC\s+\d+',                  # AIR 2012 SC 123
    r'\d{4}\s+\(\d+\)\s+SCC\s+\d+',             # 2012 (9) SCC 1
    r'Civil Appeal No\.\s*\d+',
    r'Criminal Appeal No\.\s*\d+',
    r'Writ Petition \(Civil\) No\.\s*\d+',
    r'SLP \(Civil\) No\.\s*\d+',
]

SECTION_HEADERS = [
    r'\bFACTS?\b',
    r'\bBACKGROUND\b',
    r'\bISSUES?\b',
    r'\bQUESTIONS?\s+OF\s+LAW\b',
    r'\bHELD\b',
    r'\bJUDGMENT\b',
    r'\bORDER\b',
    r'\bREASONS?\b',
    r'\bCONCLUSION\b',
    r'\bDISPOSAL\b',
]

COURT_PATTERNS = {
    "Supreme Court": r'IN\s+THE\s+SUPREME\s+COURT\s+OF\s+INDIA',
    "Delhi High Court": r'IN\s+THE\s+HIGH\s+COURT\s+OF\s+DELHI',
    "Bombay High Court": r'IN\s+THE\s+HIGH\s+COURT\s+AT\s+BOMBAY',
    "Madras High Court": r'IN\s+THE\s+HIGH\s+COURT\s+OF\s+JUDICATURE\s+AT\s+MADRAS',
    "Calcutta High Court": r'IN\s+THE\s+HIGH\s+COURT\s+AT\s+CALCUTTA',
    "Karnataka High Court": r'IN\s+THE\s+HIGH\s+COURT\s+OF\s+KARNATAKA',
}


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, int]:
    """Extract clean text from PDF, handling multi-column layouts."""
    doc = fitz.open(str(pdf_path))
    pages_text = []

    for page in doc:
        # Use dict extraction for better layout handling
        blocks = page.get_text("blocks")
        # Sort blocks by vertical then horizontal position
        blocks.sort(key=lambda b: (round(b[1] / 20) * 20, b[0]))
        page_text = "\n".join(b[4] for b in blocks if b[4].strip())
        pages_text.append(page_text)

    doc.close()
    full_text = "\n\n".join(pages_text)

    # Clean up common PDF extraction artefacts
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    full_text = re.sub(r'[ \t]+', ' ', full_text)
    full_text = re.sub(r'\x0c', '\n', full_text)  # form feeds

    return full_text, len(pages_text)


def detect_citation(text: str) -> Optional[str]:
    for pattern in CITATION_PATTERNS:
        match = re.search(pattern, text[:2000], re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


def detect_court(text: str) -> Optional[str]:
    header = text[:1000].upper()
    for court_name, pattern in COURT_PATTERNS.items():
        if re.search(pattern, header, re.IGNORECASE):
            return court_name
    return None


def detect_year(text: str) -> Optional[int]:
    # Look for year in citation or header
    match = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', text[:2000])
    if match:
        return int(match.group(1))
    return None


def detect_case_name(text: str) -> Optional[str]:
    """
    Best-effort: look for "X v. Y" or "X versus Y" pattern near top.
    Indian judgments typically have the case name in the first 500 chars.
    """
    match = re.search(
        r'([A-Z][A-Za-z\s\.]+)\s+[Vv](?:ersus|\.)\s+([A-Z][A-Za-z\s\.]+)',
        text[:1000]
    )
    if match:
        return f"{match.group(1).strip()} v. {match.group(2).strip()}"
    return None


def detect_coram(text: str) -> Optional[str]:
    """Extract bench composition from CORAM/HON'BLE section."""
    match = re.search(
        r'(?:CORAM|HON.?BLE)[:\s]+(.+?)(?=\n\n|\Z)',
        text[:3000],
        re.IGNORECASE | re.DOTALL
    )
    if match:
        coram = match.group(1).strip()
        coram = re.sub(r'\s+', ' ', coram)
        return coram[:300]  # cap length
    return None


def split_sections(text: str) -> dict[str, str]:
    """
    Attempt to split judgment into broad sections.
    Returns dict with keys: facts, issues, reasoning, held.
    All values may be None if section not detected.
    """
    sections = {}
    pattern = '|'.join(f'(?P<{i}>^{h})' for i, h in enumerate(SECTION_HEADERS))

    # Simple approach: find first occurrence of known headers
    facts_match = re.search(r'\b(FACTS?|BACKGROUND)\b', text, re.IGNORECASE | re.MULTILINE)
    held_match = re.search(r'\b(HELD|CONCLUSION|ORDER)\b', text, re.IGNORECASE | re.MULTILINE)
    issues_match = re.search(r'\b(ISSUES?|QUESTIONS?\s+OF\s+LAW)\b', text, re.IGNORECASE | re.MULTILINE)

    if facts_match and held_match and facts_match.start() < held_match.start():
        sections['facts_text'] = text[facts_match.start():held_match.start()].strip()
        sections['held_text'] = text[held_match.start():].strip()

    if issues_match:
        sections['issues_text'] = text[issues_match.start():
                                       (held_match.start() if held_match else len(text))].strip()

    return sections


def extract_judgment(pdf_path: Path) -> ExtractedJudgment:
    """Full extraction pipeline for a single PDF."""
    full_text, pages = extract_text_from_pdf(pdf_path)

    sections = split_sections(full_text)

    return ExtractedJudgment(
        filename=pdf_path.name,
        full_text=full_text,
        pages=pages,
        case_name=detect_case_name(full_text),
        citation=detect_citation(full_text),
        court=detect_court(full_text),
        year=detect_year(full_text),
        coram=detect_coram(full_text),
        **sections
    )


def process_file(pdf_path: Path, output_dir: Path) -> Path:
    """Extract one PDF and write JSON to output_dir."""
    judgment = extract_judgment(pdf_path)
    out_path = output_dir / (pdf_path.stem + "_extracted.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(judgment), f, ensure_ascii=False, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from Indian judgment PDFs"
    )
    parser.add_argument("--input", required=True,
                        help="Path to PDF file or directory of PDFs")
    parser.add_argument("--output", required=True,
                        help="Output directory for extracted JSON files")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        pdf_files = [input_path]
    elif input_path.is_dir():
        pdf_files = list(input_path.glob("**/*.pdf"))
        console.print(f"Found [bold]{len(pdf_files)}[/bold] PDF files")
    else:
        console.print(f"[red]Input path not found: {input_path}[/red]")
        return

    results = {"success": 0, "failed": 0}

    for pdf_path in tqdm(pdf_files, desc="Extracting"):
        try:
            out_path = process_file(pdf_path, output_dir)
            results["success"] += 1
            if len(pdf_files) == 1:
                console.print(f"[green]Extracted:[/green] {out_path}")
        except Exception as e:
            results["failed"] += 1
            console.print(f"[red]Failed:[/red] {pdf_path.name}: {e}")

    console.print(f"\n[green]Done.[/green] {results['success']} extracted, "
                  f"{results['failed']} failed.")
    console.print(f"Next step: python ingestion/irac_parser.py --input {output_dir}")


if __name__ == "__main__":
    main()
