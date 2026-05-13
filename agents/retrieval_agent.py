"""
agents/retrieval_agent.py

Retrieves precedent candidates and reasoning paths from FalkorDB
given a natural language legal query.

The retrieval agent does NOT generate text. It traverses the graph
and returns structured candidate paths for the LLM synthesis step.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from falkordb import FalkorDB

load_dotenv()


@dataclass
class PrecedentCandidate:
    citation: str
    case_name: str
    court: str
    year: int
    matter_type: str
    summary: str
    relevance_reason: str = ""       # why this case was retrieved
    relationship_to_query: str = ""  # CITES | APPLIES_RULE | DECIDES


@dataclass
class RetrievalResult:
    query: str
    candidates: list[PrecedentCandidate] = field(default_factory=list)
    statutes: list[dict] = field(default_factory=list)
    procedural_path: list[str] = field(default_factory=list)
    conflict_flags: list[dict] = field(default_factory=list)


class RetrievalAgent:
    """
    Traverses FalkorDB to find relevant precedents, statutes,
    and procedural paths for a given legal query.
    """

    def __init__(self, graph_url: str = None, graph_name: str = None):
        host = os.getenv("FALKORDB_HOST", "localhost")
        port = int(os.getenv("FALKORDB_PORT", 6381))
        graph_name = graph_name or os.getenv("FALKORDB_GRAPH_NAME", "legal_graph")

        db = FalkorDB(host=host, port=port)
        self.graph = db.select_graph(graph_name)

    def run(self, query: str) -> list:
        result = self.graph.query(query)
        return result.result_set

    # ----------------------------------------------------------------
    # Graph queries
    # ----------------------------------------------------------------

    def find_cases_by_matter_type(self, matter_type: str, limit: int = 10) -> list[PrecedentCandidate]:
        rows = self.run(f"""
            MATCH (c:Case)
            WHERE c.matter_type = '{matter_type}'
            RETURN c.citation, c.name, c.court, c.year, c.matter_type, c.summary
            ORDER BY c.year DESC
            LIMIT {limit}
        """)
        return [
            PrecedentCandidate(
                citation=r[0] or "",
                case_name=r[1] or "",
                court=r[2] or "",
                year=r[3] or 0,
                matter_type=r[4] or "",
                summary=r[5] or "",
                relevance_reason=f"matter_type match: {matter_type}"
            )
            for r in rows
        ]

    def find_cases_citing_section(self, statute_name: str, section_number: str) -> list[PrecedentCandidate]:
        rows = self.run(f"""
            MATCH (c:Case)-[:CITES_STATUTE]->(sec:Section {{number: '{section_number}', statute: '{statute_name}'}})
            RETURN c.citation, c.name, c.court, c.year, c.matter_type, c.summary
            ORDER BY c.year DESC
            LIMIT 15
        """)
        return [
            PrecedentCandidate(
                citation=r[0] or "",
                case_name=r[1] or "",
                court=r[2] or "",
                year=r[3] or 0,
                matter_type=r[4] or "",
                summary=r[5] or "",
                relevance_reason=f"cites {statute_name} s.{section_number}"
            )
            for r in rows
        ]

    def find_cases_by_legal_issue(self, issue_keywords: list[str]) -> list[PrecedentCandidate]:
        """Find cases that decided issues containing any of the keywords."""
        keyword_conditions = " OR ".join(
            f"i.text CONTAINS '{kw}'" for kw in issue_keywords
        )
        rows = self.run(f"""
            MATCH (c:Case)-[:DECIDES]->(i:LegalIssue)
            WHERE {keyword_conditions}
            RETURN c.citation, c.name, c.court, c.year, c.matter_type, c.summary, i.text
            ORDER BY c.year DESC
            LIMIT 15
        """)
        return [
            PrecedentCandidate(
                citation=r[0] or "",
                case_name=r[1] or "",
                court=r[2] or "",
                year=r[3] or 0,
                matter_type=r[4] or "",
                summary=r[5] or "",
                relevance_reason=f"decided issue: {r[6]}"
            )
            for r in rows
        ]

    def find_citation_chain(self, citation: str, depth: int = 2) -> list[dict]:
        """Traverse the citation graph outward from a given case."""
        rows = self.run(f"""
            MATCH path = (c:Case {{citation: '{citation}'}})-[:CITES*1..{depth}]->(p:Case)
            RETURN p.citation, p.name, p.year, length(path) as depth
            ORDER BY depth, p.year DESC
        """)
        return [
            {"citation": r[0], "name": r[1], "year": r[2], "depth": r[3]}
            for r in rows
        ]

    def find_conflicts(self, citation: str) -> list[dict]:
        """Find any CONFLICTS_WITH relationships for a given case."""
        rows = self.run(f"""
            MATCH (c:Case {{citation: '{citation}'}})-[r:CONFLICTS_WITH]->(other:Case)
            RETURN other.citation, other.name, r.conflict_type, r.unresolved
        """)
        conflicts = []
        for r in rows:
            conflict = {
                "conflicting_citation": r[0],
                "conflicting_name": r[1],
                "conflict_type": r[2],
                "unresolved": r[3],
                "resolved_by": None
            }
            # Check if there is a resolution
            res_rows = self.run(f"""
                MATCH (c:Case {{citation: '{citation}'}})-[:RESOLVED_BY]->(res:Case)
                RETURN res.citation, res.name
                LIMIT 1
            """)
            if res_rows:
                conflict["resolved_by"] = {
                    "citation": res_rows[0][0],
                    "name": res_rows[0][1]
                }
            conflicts.append(conflict)
        return conflicts

    def find_procedural_path(self, event_type: str) -> list[str]:
        """
        Given a procedural event type, return the full expected
        sequence of events before and after it.
        """
        rows = self.run(f"""
            MATCH path = (start:ProceduralEvent)-[:PRECEDES*]->(e:ProceduralEvent {{event_type: '{event_type}'}})
            WHERE NOT ()-[:PRECEDES]->(start)
            RETURN [node IN nodes(path) | node.event_type] AS event_sequence
            LIMIT 1
        """)
        if rows:
            return rows[0][0]

        # Fall back: just find what comes after this event
        rows = self.run(f"""
            MATCH (e:ProceduralEvent {{event_type: '{event_type}'}})-[:PRECEDES*]->(next:ProceduralEvent)
            RETURN next.event_type
            ORDER BY next.event_type
            LIMIT 5
        """)
        if rows:
            return [event_type] + [r[0] for r in rows]

        return [event_type]

    def find_statutes_for_matter(self, matter_type: str) -> list[dict]:
        """Find commonly cited statutes for a given matter type."""
        rows = self.run(f"""
            MATCH (c:Case {{matter_type: '{matter_type}'}})-[:CITES_STATUTE]->(sec:Section)-[:PART_OF]->(s:Statute)
            RETURN s.name, sec.number, count(*) as freq
            ORDER BY freq DESC
            LIMIT 10
        """)
        return [
            {"statute": r[0], "section": r[1], "frequency": r[2]}
            for r in rows
        ]

    # ----------------------------------------------------------------
    # Main retrieval entry point
    # ----------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        matter_type: str = None,
        statute_refs: list[dict] = None,
        issue_keywords: list[str] = None,
        limit: int = 10
    ) -> RetrievalResult:
        """
        Main retrieval method. Aggregates candidates from multiple
        graph traversal strategies and deduplicates by citation.
        """
        result = RetrievalResult(query=query)
        seen_citations = set()

        def add_candidates(candidates: list[PrecedentCandidate]):
            for c in candidates:
                if c.citation and c.citation not in seen_citations:
                    seen_citations.add(c.citation)
                    result.candidates.append(c)

        # 1. Matter type traversal
        if matter_type:
            add_candidates(self.find_cases_by_matter_type(matter_type, limit=limit))
            result.statutes = self.find_statutes_for_matter(matter_type)

        # 2. Statute/section traversal
        if statute_refs:
            for ref in statute_refs:
                add_candidates(self.find_cases_citing_section(
                    ref.get("statute_name", ""),
                    ref.get("section_number", "")
                ))

        # 3. Issue keyword traversal
        if issue_keywords:
            add_candidates(self.find_cases_by_legal_issue(issue_keywords))

        # 4. Check for conflicts among retrieved candidates
        for candidate in result.candidates[:5]:  # check top 5 only
            conflicts = self.find_conflicts(candidate.citation)
            if conflicts:
                result.conflict_flags.extend(conflicts)

        # 5. Rank: Supreme Court > High Court, newer > older
        result.candidates = sorted(
            result.candidates,
            key=lambda c: (
                1 if "Supreme" in c.court else 0,
                c.year
            ),
            reverse=True
        )[:limit]

        return result
