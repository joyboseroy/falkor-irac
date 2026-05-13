"""
ingestion/irac_parser.py

Uses an LLM to parse extracted judgment text into structured IRAC nodes
suitable for loading into FalkorDB.

Input:  JSON files from pdf_extractor.py
Output: JSON files with structured IRAC graph data

Usage:
    python ingestion/irac_parser.py --input data/processed/judgment_extracted.json
    python ingestion/irac_parser.py --input data/processed/ --batch
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm
from rich.console import Console
from rich.pretty import pprint

load_dotenv()
console = Console()

# ----------------------------------------------------------------
# Data models for IRAC graph nodes
# ----------------------------------------------------------------

@dataclass
class CaseNode:
    citation: str
    name: str
    court: str
    year: int
    bench_size: int = 2
    bench_type: str = "division"
    matter_type: str = "general"
    summary: str = ""
    ildc_id: str = ""


@dataclass
class LegalIssueNode:
    text: str
    issue_type: str = "substantive"   # constitutional|procedural|substantive|evidentiary


@dataclass
class RuleNode:
    text: str
    source: str = "precedent"         # precedent|statute|custom


@dataclass
class StatuteRef:
    statute_name: str
    section_number: str
    purpose: str = "relied_upon"      # relied_upon|distinguished|referred


@dataclass
class PrecedentRef:
    citation: str
    case_name: str
    relationship: str = "CITES"       # CITES|OVERRULES|DISTINGUISHES|CONFLICTS_WITH
    proposition: str = ""
    conflict_type: str = ""           # coordinate_bench|per_incuriam|distinguished


@dataclass
class IRACStructure:
    """
    Full IRAC representation of a single judgment,
    ready for loading into FalkorDB.
    """
    case: CaseNode
    issues: list[LegalIssueNode] = field(default_factory=list)
    rules: list[RuleNode] = field(default_factory=list)
    analysis_summary: str = ""
    conclusion: str = ""
    outcome_type: str = "dismissed"   # allowed|dismissed|modified|remanded
    statutes_cited: list[StatuteRef] = field(default_factory=list)
    precedents_cited: list[PrecedentRef] = field(default_factory=list)
    procedural_events: list[str] = field(default_factory=list)
    extraction_confidence: float = 0.0
    extraction_notes: str = ""


# ----------------------------------------------------------------
# LLM prompt
# ----------------------------------------------------------------

SYSTEM_PROMPT = "Return only valid JSON. No explanation. No markdown."

EXTRACTION_PROMPT = """From the Indian court judgment below, extract key fields as JSON.
Return ONLY this JSON, nothing else:
{{"case":{{"citation":"","name":"","court":"","year":0,"bench_size":2,"bench_type":"division","matter_type":"general","summary":""}},"issues":[{{"text":"","issue_type":"substantive"}}],"rules":[{{"text":"","source":"precedent"}}],"analysis_summary":"","conclusion":"","outcome_type":"dismissed","statutes_cited":[{{"statute_name":"","section_number":"","purpose":"relied_upon"}}],"precedents_cited":[{{"citation":"","case_name":"","relationship":"CITES","proposition":"","conflict_type":""}}],"procedural_events":[],"extraction_confidence":0.5,"extraction_notes":""}}

JUDGMENT:
{judgment_text}
"""


def truncate_for_context(text: str, max_chars: int = 600) -> str:
    """
    Truncate judgment text to fit TinyLlama's context window.
    Uses only the first portion where case name, citation and key facts appear.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def call_llm(prompt: str, model: str = None) -> dict:
    """Call Ollama and return parsed JSON dict. No API key required."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agents.llm import call_llm_json
    return call_llm_json(prompt, system=SYSTEM_PROMPT)


def parse_llm_response(raw_or_dict) -> dict:
    """Accept either a dict (from Ollama path) or a raw string."""
    if isinstance(raw_or_dict, dict):
        return raw_or_dict
    raw = raw_or_dict.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return json.loads(raw)


def dict_to_irac(data: dict) -> IRACStructure:
    """Convert raw LLM JSON dict to typed IRACStructure."""
    case_data = data.get("case", {})
    case = CaseNode(
        citation=case_data.get("citation", ""),
        name=case_data.get("name", ""),
        court=case_data.get("court", ""),
        year=int(case_data.get("year", 0)),
        bench_size=int(case_data.get("bench_size", 2)),
        bench_type=case_data.get("bench_type", "division"),
        matter_type=case_data.get("matter_type", "general"),
        summary=case_data.get("summary", ""),
    )

    issues = [
        LegalIssueNode(
            text=i.get("text", ""),
            issue_type=i.get("issue_type", "substantive")
        )
        for i in data.get("issues", [])
        if i.get("text")
    ]

    rules = [
        RuleNode(
            text=r.get("text", ""),
            source=r.get("source", "precedent")
        )
        for r in data.get("rules", [])
        if r.get("text")
    ]

    statutes = [
        StatuteRef(
            statute_name=s.get("statute_name", ""),
            section_number=s.get("section_number", ""),
            purpose=s.get("purpose", "relied_upon")
        )
        for s in data.get("statutes_cited", [])
        if s.get("statute_name")
    ]

    precedents = [
        PrecedentRef(
            citation=p.get("citation", ""),
            case_name=p.get("case_name", ""),
            relationship=p.get("relationship", "CITES"),
            proposition=p.get("proposition", ""),
            conflict_type=p.get("conflict_type", "")
        )
        for p in data.get("precedents_cited", [])
        if p.get("case_name")
    ]

    return IRACStructure(
        case=case,
        issues=issues,
        rules=rules,
        analysis_summary=data.get("analysis_summary", ""),
        conclusion=data.get("conclusion", ""),
        outcome_type=data.get("outcome_type", "dismissed"),
        statutes_cited=statutes,
        precedents_cited=precedents,
        procedural_events=data.get("procedural_events", []),
        extraction_confidence=float(data.get("extraction_confidence", 0.0)),
        extraction_notes=data.get("extraction_notes", ""),
    )


def parse_judgment(extracted_path: Path) -> IRACStructure:
    """Full pipeline: load extracted JSON, call LLM, return IRACStructure."""
    with open(extracted_path, encoding="utf-8") as f:
        extracted = json.load(f)

    # Use section-split text if available, else full text
    judgment_text = (
        "\n\n".join(filter(None, [
            extracted.get("facts_text"),
            extracted.get("issues_text"),
            extracted.get("held_text"),
        ]))
        or extracted.get("full_text", "")
    )

    judgment_text = truncate_for_context(judgment_text)
    prompt = EXTRACTION_PROMPT.format(judgment_text=judgment_text)

    raw_response = call_llm(prompt)
    data = parse_llm_response(raw_response)
    irac = dict_to_irac(data)

    # Backfill from PDF extraction if LLM left fields empty
    if not irac.case.citation and extracted.get("citation"):
        irac.case.citation = extracted["citation"]
    if not irac.case.name and extracted.get("case_name"):
        irac.case.name = extracted["case_name"]
    if not irac.case.court and extracted.get("court"):
        irac.case.court = extracted["court"]
    if not irac.case.year and extracted.get("year"):
        irac.case.year = extracted["year"]

    return irac


def process_file(input_path: Path, output_dir: Path, delay: float = 1.0) -> Path:
    """Parse one extracted JSON and write IRAC JSON to output_dir."""
    irac = parse_judgment(input_path)
    stem = input_path.stem.replace("_extracted", "")
    out_path = output_dir / (stem + "_irac.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(irac), f, ensure_ascii=False, indent=2)
    time.sleep(delay)  # rate limiting for API calls
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="LLM-assisted IRAC extraction from judgment text"
    )
    parser.add_argument("--input", required=True,
                        help="Path to extracted JSON file or directory")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between API calls in batch mode (default: 1.0)")
    parser.add_argument("--batch", action="store_true",
                        help="Process all *_extracted.json files in directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(input_path.glob("**/*_extracted.json"))
        console.print(f"Found [bold]{len(files)}[/bold] extracted JSON files")
    else:
        console.print(f"[red]Input path not found: {input_path}[/red]")
        return

    results = {"success": 0, "failed": 0}

    for f in tqdm(files, desc="Parsing IRAC"):
        try:
            out_path = process_file(f, output_dir, delay=args.delay)
            results["success"] += 1
            if len(files) == 1:
                console.print(f"[green]Parsed:[/green] {out_path}")
                with open(out_path) as jf:
                    data = json.load(jf)
                console.print(f"  Case: [bold]{data['case']['name']}[/bold]")
                console.print(f"  Issues: {len(data['issues'])}")
                console.print(f"  Precedents cited: {len(data['precedents_cited'])}")
                console.print(f"  Confidence: {data['extraction_confidence']:.2f}")
        except Exception as e:
            results["failed"] += 1
            console.print(f"[red]Failed:[/red] {f.name}: {e}")

    console.print(f"\n[green]Done.[/green] {results['success']} parsed, "
                  f"{results['failed']} failed.")
    console.print(f"Next step: python ingestion/graph_loader.py --input {output_dir}")


if __name__ == "__main__":
    main()
