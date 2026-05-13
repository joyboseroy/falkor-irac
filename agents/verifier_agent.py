"""
agents/verifier_agent.py

The Verifier Agent: a falsifiability oracle.

Given a proposed answer and the citation chain it claims to rely on,
the verifier checks whether a valid supporting path exists in the
FalkorDB graph. It does NOT generate text.

Output:
    VALID    -- citation path exists, statutes are current
    INVALID  -- citation path does not exist in graph (hallucination risk)
    CONFLICT -- multiple valid but contradictory paths found
    STALE    -- cited statute or section has been repealed
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from falkordb import FalkorDB

load_dotenv()


class VerificationStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    CONFLICT = "CONFLICT"
    STALE = "STALE"


@dataclass
class CitationCheck:
    citation: str
    exists_in_graph: bool
    court: str = ""
    year: int = 0
    is_overruled: bool = False
    overruled_by: str = ""


@dataclass
class StatuteCheck:
    statute_name: str
    section_number: str
    exists_in_graph: bool
    is_repealed: bool = False
    repealed_by: str = ""


@dataclass
class VerificationResult:
    status: VerificationStatus
    citation_checks: list[CitationCheck] = field(default_factory=list)
    statute_checks: list[StatuteCheck] = field(default_factory=list)
    conflict_details: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""

    def to_output_dict(self) -> dict:
        return {
            "status": self.status.value,
            "confidence": self.confidence,
            "notes": self.notes,
            "citations_verified": [
                {
                    "citation": c.citation,
                    "exists": c.exists_in_graph,
                    "is_overruled": c.is_overruled,
                    "overruled_by": c.overruled_by,
                }
                for c in self.citation_checks
            ],
            "statutes_verified": [
                {
                    "statute": s.statute_name,
                    "section": s.section_number,
                    "exists": s.exists_in_graph,
                    "is_repealed": s.is_repealed,
                    "repealed_by": s.repealed_by,
                }
                for s in self.statute_checks
            ],
            "conflicts": self.conflict_details,
        }


class VerifierAgent:
    """
    Checks whether a proposed legal answer can be grounded
    to valid paths in the FalkorDB legal knowledge graph.

    This is the anti-hallucination mechanism:
        LLM proposes -> Verifier checks path existence -> Accept or Reject
    """

    def __init__(
        self,
        graph_url: str = None,
        graph_name: str = None,
        requires_citation_path: bool = True
    ):
        host = os.getenv("FALKORDB_HOST", "localhost")
        port = int(os.getenv("FALKORDB_PORT", 6381))
        graph_name = graph_name or os.getenv("FALKORDB_GRAPH_NAME", "legal_graph")
        self.requires_citation_path = requires_citation_path

        db = FalkorDB(host=host, port=port)
        self.graph = db.select_graph(graph_name)

    def run(self, query: str) -> list:
        result = self.graph.query(query)
        return result.result_set

    # ----------------------------------------------------------------
    # Individual checks
    # ----------------------------------------------------------------

    def check_citation_exists(self, citation: str) -> CitationCheck:
        """Check whether a cited case exists in the graph."""
        citation_clean = citation.replace("'", "\\'")
        rows = self.run(f"""
            MATCH (c:Case {{citation: '{citation_clean}'}})
            RETURN c.court, c.year
            LIMIT 1
        """)

        if not rows:
            # Try fuzzy match on name
            rows = self.run(f"""
                MATCH (c:Case)
                WHERE c.name CONTAINS '{citation_clean}'
                RETURN c.court, c.year
                LIMIT 1
            """)

        if not rows:
            return CitationCheck(citation=citation, exists_in_graph=False)

        court = rows[0][0] or ""
        year = rows[0][1] or 0

        # Check if the case has been overruled
        overruled_rows = self.run(f"""
            MATCH (later:Case)-[:OVERRULES]->(c:Case {{citation: '{citation_clean}'}})
            RETURN later.citation
            LIMIT 1
        """)

        overruled_by = overruled_rows[0][0] if overruled_rows else ""

        return CitationCheck(
            citation=citation,
            exists_in_graph=True,
            court=court,
            year=year,
            is_overruled=bool(overruled_by),
            overruled_by=overruled_by
        )

    def check_statute_current(self, statute_name: str, section_number: str = "") -> StatuteCheck:
        """Check whether a cited statute or section is current (not repealed)."""
        statute_clean = statute_name.replace("'", "\\'")

        if section_number:
            section_clean = section_number.replace("'", "\\'")
            rows = self.run(f"""
                MATCH (sec:Section {{number: '{section_clean}', statute: '{statute_clean}'}})
                RETURN sec.repealed, sec.repealed_by
                LIMIT 1
            """)
            if rows:
                return StatuteCheck(
                    statute_name=statute_name,
                    section_number=section_number,
                    exists_in_graph=True,
                    is_repealed=bool(rows[0][0]),
                    repealed_by=rows[0][1] or ""
                )

        # Check at statute level
        rows = self.run(f"""
            MATCH (s:Statute {{name: '{statute_clean}'}})
            RETURN s.repealed, s.repealed_year
            LIMIT 1
        """)
        if rows:
            return StatuteCheck(
                statute_name=statute_name,
                section_number=section_number,
                exists_in_graph=True,
                is_repealed=bool(rows[0][0]),
                repealed_by=str(rows[0][1]) if rows[0][1] else ""
            )

        return StatuteCheck(
            statute_name=statute_name,
            section_number=section_number,
            exists_in_graph=False
        )

    def check_for_conflicts(self, citations: list[str]) -> list[dict]:
        """
        Given a set of citations proposed in the same answer,
        check whether any pair conflicts with each other.
        """
        conflicts = []
        for i, cit_a in enumerate(citations):
            for cit_b in citations[i+1:]:
                a_clean = cit_a.replace("'", "\\'")
                b_clean = cit_b.replace("'", "\\'")
                rows = self.run(f"""
                    MATCH (a:Case {{citation: '{a_clean}'}})-[r:CONFLICTS_WITH]->(b:Case {{citation: '{b_clean}'}})
                    RETURN r.conflict_type, r.unresolved
                    LIMIT 1
                """)
                if rows:
                    conflict = {
                        "case_a": cit_a,
                        "case_b": cit_b,
                        "conflict_type": rows[0][0],
                        "unresolved": rows[0][1],
                    }
                    # Check for resolution
                    res_rows = self.run(f"""
                        MATCH (a:Case {{citation: '{a_clean}'}})-[:RESOLVED_BY]->(res:Case)
                        RETURN res.citation, res.name
                        LIMIT 1
                    """)
                    if res_rows:
                        conflict["resolved_by"] = {
                            "citation": res_rows[0][0],
                            "name": res_rows[0][1]
                        }
                        conflict["unresolved"] = False
                    else:
                        conflict["resolved_by"] = None
                    conflicts.append(conflict)
        return conflicts

    # ----------------------------------------------------------------
    # Main verification entry point
    # ----------------------------------------------------------------

    def check(
        self,
        proposed_answer: str,
        citations: list[str] = None,
        statute_refs: list[dict] = None,
    ) -> VerificationResult:
        """
        Full verification check.

        Args:
            proposed_answer: the LLM-generated answer text (used for logging only)
            citations: list of case citations the answer claims to rely on
            statute_refs: list of {"statute_name": ..., "section_number": ...} dicts

        Returns:
            VerificationResult with status VALID | INVALID | CONFLICT | STALE
        """
        citations = citations or []
        statute_refs = statute_refs or []

        citation_checks = [self.check_citation_exists(c) for c in citations]
        statute_checks = [
            self.check_statute_current(
                s.get("statute_name", ""),
                s.get("section_number", "")
            )
            for s in statute_refs
        ]

        # Conflict check across proposed citations
        conflict_details = self.check_for_conflicts(citations) if len(citations) > 1 else []

        # Determine status
        missing = [c for c in citation_checks if not c.exists_in_graph]
        overruled = [c for c in citation_checks if c.is_overruled]
        stale_statutes = [s for s in statute_checks if s.is_repealed]
        unresolved_conflicts = [cf for cf in conflict_details if cf.get("unresolved")]

        if stale_statutes:
            status = VerificationStatus.STALE
            notes = (
                f"Cited statute(s) may be repealed: "
                f"{', '.join(s.statute_name for s in stale_statutes)}"
            )
        elif unresolved_conflicts:
            status = VerificationStatus.CONFLICT
            notes = (
                f"Unresolved doctrinal conflict between: "
                f"{', '.join(cf['case_a'] + ' vs ' + cf['case_b'] for cf in unresolved_conflicts)}"
            )
        elif missing and self.requires_citation_path:
            status = VerificationStatus.INVALID
            notes = (
                f"Citations not found in graph: "
                f"{', '.join(c.citation for c in missing)}. "
                "Possible hallucination."
            )
        elif overruled:
            status = VerificationStatus.INVALID
            notes = (
                f"Cited case(s) have been overruled: "
                f"{', '.join(c.citation + ' by ' + c.overruled_by for c in overruled)}"
            )
        else:
            status = VerificationStatus.VALID
            notes = "All citations verified in graph."

        # Confidence: proportion of citations verified
        if citations:
            verified_count = sum(1 for c in citation_checks if c.exists_in_graph and not c.is_overruled)
            confidence = verified_count / len(citations)
        else:
            confidence = 0.5  # no citations to verify

        return VerificationResult(
            status=status,
            citation_checks=citation_checks,
            statute_checks=statute_checks,
            conflict_details=conflict_details,
            confidence=confidence,
            notes=notes
        )
