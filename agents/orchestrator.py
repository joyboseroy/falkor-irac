"""
agents/orchestrator.py

Graph-Constrained Generation pipeline.

Architecture:
    User Query
        -> RetrievalAgent (graph traversal, precedent candidates)
        -> LLM Synthesis  (path-guided answer generation)
        -> VerifierAgent  (falsifiability oracle)
        -> Output: answer + citation chain + verification status

The LLM proposes; the Verifier has veto power.
On INVALID or STALE: the LLM is prompted to revise or abstain.
On CONFLICT: the conflict is surfaced as first-class output.
"""

import json
import os
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Allow running as: python agents/orchestrator.py from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from agents.retrieval_agent import RetrievalAgent, RetrievalResult
from agents.verifier_agent import VerifierAgent, VerificationStatus, VerificationResult

load_dotenv()

MAX_REVISION_ATTEMPTS = 2


@dataclass
class LegalAnswer:
    """Final output of the orchestrator."""
    query: str
    answer: str
    citations: list[str]
    statute_refs: list[dict]
    verification: VerificationResult
    conflict_output: Optional[dict] = None
    revision_attempts: int = 0
    abstained: bool = False

    def to_dict(self) -> dict:
        out = {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "statute_refs": self.statute_refs,
            "verification": self.verification.to_output_dict(),
            "revision_attempts": self.revision_attempts,
            "abstained": self.abstained,
        }
        if self.conflict_output:
            out["conflict"] = self.conflict_output
        return out


# ----------------------------------------------------------------
# LLM synthesis prompts
# ----------------------------------------------------------------

SYNTHESIS_SYSTEM = """You are a legal research assistant specialising in Indian law.
You answer queries about Indian court judgments, statutes, and legal procedure.
You must base your answer strictly on the provided precedents and statutes.
Do not cite any case not listed in the provided candidates.
Return your response as a JSON object only. No preamble, no markdown.
"""

SYNTHESIS_PROMPT = """Answer the following legal query using only the provided precedents and statutes.

QUERY: {query}

RELEVANT PRECEDENTS:
{precedents}

RELEVANT STATUTES:
{statutes}

Return a JSON object with this structure:
{{
  "answer": "string - your legal answer in plain language",
  "citations": ["list of case citations you are relying on"],
  "statute_refs": [
    {{"statute_name": "string", "section_number": "string"}}
  ],
  "reasoning": "string - brief explanation of your reasoning chain"
}}

Restrict citations to those in the provided precedents list. Do not invent citations.
"""

REVISION_PROMPT = """Your previous answer was rejected for the following reason:
{rejection_reason}

Please revise your answer using only the provided precedents.
If you cannot answer reliably from the provided materials, set answer to
"I cannot provide a verified answer to this query from available precedents."
and set citations to an empty list.

ORIGINAL QUERY: {query}

AVAILABLE PRECEDENTS:
{precedents}

Return the same JSON structure as before.
"""


def format_precedents(retrieval: RetrievalResult) -> str:
    lines = []
    for c in retrieval.candidates:
        lines.append(
            f"- {c.case_name} [{c.citation}] ({c.court}, {c.year})\n"
            f"  Summary: {c.summary[:200]}\n"
            f"  Relevance: {c.relevance_reason}"
        )
    return "\n".join(lines) or "No precedents found."


def format_statutes(retrieval: RetrievalResult) -> str:
    lines = []
    for s in retrieval.statutes:
        lines.append(f"- {s['statute']} s.{s['section']} (cited {s['frequency']} times)")
    return "\n".join(lines) or "No statutes found."


def call_llm(prompt: str, system: str = SYNTHESIS_SYSTEM) -> dict:
    """Call Ollama/tinyllama. No API key required."""
    from agents.llm import call_llm_json
    return call_llm_json(prompt, system=system)


# ----------------------------------------------------------------
# Main orchestrator
# ----------------------------------------------------------------

class Orchestrator:
    """
    Graph-Constrained Generation pipeline.

    Usage:
        orchestrator = Orchestrator()
        result = orchestrator.answer(
            query="What are the grounds for bail denial in India?",
            matter_type="bail",
            issue_keywords=["bail", "denial", "grounds"]
        )
        print(result.to_dict())
    """

    def __init__(self):
        self.retriever = RetrievalAgent()
        self.verifier = VerifierAgent(requires_citation_path=True)

    def answer(
        self,
        query: str,
        matter_type: str = None,
        statute_refs: list[dict] = None,
        issue_keywords: list[str] = None,
        limit: int = 8,
    ) -> LegalAnswer:
        """
        Full pipeline: retrieve -> synthesize -> verify -> (revise if needed).
        """

        # Step 1: Retrieve candidate precedents from graph
        retrieval = self.retriever.retrieve(
            query=query,
            matter_type=matter_type,
            statute_refs=statute_refs,
            issue_keywords=issue_keywords,
            limit=limit,
        )

        precedents_text = format_precedents(retrieval)
        statutes_text = format_statutes(retrieval)

        # Step 2: LLM synthesis
        synthesis_prompt = SYNTHESIS_PROMPT.format(
            query=query,
            precedents=precedents_text,
            statutes=statutes_text,
        )

        llm_output = call_llm(synthesis_prompt)
        citations = llm_output.get("citations", [])
        statute_refs_out = llm_output.get("statute_refs", [])

        # Step 3: Verify
        verification = self.verifier.check(
            proposed_answer=llm_output.get("answer", ""),
            citations=citations,
            statute_refs=statute_refs_out,
        )

        revision_attempts = 0

        # Step 4: Revise if INVALID or STALE (up to MAX_REVISION_ATTEMPTS)
        while (
            verification.status in (VerificationStatus.INVALID, VerificationStatus.STALE)
            and revision_attempts < MAX_REVISION_ATTEMPTS
        ):
            revision_attempts += 1
            revision_prompt = REVISION_PROMPT.format(
                rejection_reason=verification.notes,
                query=query,
                precedents=precedents_text,
            )
            llm_output = call_llm(revision_prompt)
            citations = llm_output.get("citations", [])
            statute_refs_out = llm_output.get("statute_refs", [])
            verification = self.verifier.check(
                proposed_answer=llm_output.get("answer", ""),
                citations=citations,
                statute_refs=statute_refs_out,
            )

        answer_text = llm_output.get("answer", "")
        abstained = (
            "cannot provide a verified answer" in answer_text.lower()
            or not citations
        )

        # Step 5: Build conflict output if CONFLICT status
        conflict_output = None
        if verification.status == VerificationStatus.CONFLICT:
            conflict_output = {
                "answer": answer_text,
                "supporting_paths": citations,
                "conflict": True,
                "conflict_details": verification.conflict_details,
                "resolution": (
                    "unresolved - refer to larger bench ruling if available"
                    if any(cf.get("unresolved") for cf in verification.conflict_details)
                    else "resolved - see resolved_by field"
                ),
                "confidence": "low",
            }

        return LegalAnswer(
            query=query,
            answer=answer_text,
            citations=citations,
            statute_refs=statute_refs_out,
            verification=verification,
            conflict_output=conflict_output,
            revision_attempts=revision_attempts,
            abstained=abstained,
        )


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from rich.console import Console
    from rich.pretty import pprint

    console = Console()
    parser = argparse.ArgumentParser(description="Query the falkor-irac system")
    parser.add_argument("--query", required=True, help="Legal query")
    parser.add_argument("--matter-type", default=None,
                        help="Matter type: bail | service | constitutional | criminal | civil")
    parser.add_argument("--keywords", default=None,
                        help="Comma-separated issue keywords")
    args = parser.parse_args()

    keywords = args.keywords.split(",") if args.keywords else None

    orchestrator = Orchestrator()
    result = orchestrator.answer(
        query=args.query,
        matter_type=args.matter_type,
        issue_keywords=keywords,
    )

    console.print("\n[bold blue]QUERY[/bold blue]")
    console.print(result.query)

    console.print("\n[bold blue]ANSWER[/bold blue]")
    console.print(result.answer)

    console.print("\n[bold blue]CITATIONS[/bold blue]")
    for c in result.citations:
        console.print(f"  {c}")

    status_colour = {
        "VALID": "green", "INVALID": "red",
        "CONFLICT": "yellow", "STALE": "orange3"
    }.get(result.verification.status.value, "white")

    console.print(f"\n[bold]VERIFICATION STATUS:[/bold] [{status_colour}]{result.verification.status.value}[/{status_colour}]")
    console.print(f"Confidence: {result.verification.confidence:.2f}")
    console.print(f"Notes: {result.verification.notes}")

    if result.conflict_output:
        console.print("\n[bold yellow]CONFLICT DETECTED[/bold yellow]")
        pprint(result.conflict_output)

    if result.abstained:
        console.print("\n[yellow]System abstained: insufficient verified precedents.[/yellow]")
