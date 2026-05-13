"""
ingestion/graph_loader.py

Loads structured IRAC JSON files into FalkorDB.
Creates nodes and relationships according to the falkor-irac schema.

Usage:
    python ingestion/graph_loader.py --input data/processed/judgment_irac.json
    python ingestion/graph_loader.py --input data/processed/  # batch
    python ingestion/graph_loader.py --input data/processed/ --clear  # clear graph first
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from falkordb import FalkorDB
from tqdm import tqdm
from rich.console import Console

load_dotenv()
console = Console()


def get_graph():
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", 6381))
    graph_name = os.getenv("FALKORDB_GRAPH_NAME", "legal_graph")
    db = FalkorDB(host=host, port=port)
    return db.select_graph(graph_name)


def escape(s: str) -> str:
    """Escape single quotes for Cypher string literals."""
    if not s:
        return ""
    return s.replace("\\", "\\\\").replace("'", "\\'")


class GraphLoader:
    def __init__(self, graph):
        self.graph = graph
        self.stats = {
            "cases": 0, "issues": 0, "rules": 0,
            "statutes": 0, "sections": 0,
            "precedent_rels": 0, "procedural_events": 0,
        }

    def run(self, query: str, params: dict = None):
        return self.graph.query(query, params)

    # ----------------------------------------------------------------
    # Node upserts (MERGE ensures idempotency)
    # ----------------------------------------------------------------

    def upsert_case(self, case: dict) -> str:
        """Upsert a Case node. Returns the citation as ID."""
        citation = escape(case.get("citation", ""))
        name = escape(case.get("name", ""))
        court = escape(case.get("court", ""))
        year = int(case.get("year", 0))
        bench_size = int(case.get("bench_size", 2))
        bench_type = escape(case.get("bench_type", "division"))
        matter_type = escape(case.get("matter_type", "general"))
        summary = escape(case.get("summary", ""))

        if not citation:
            # Fall back to name as identifier
            citation = name

        self.run(f"""
            MERGE (c:Case {{citation: '{citation}'}})
            SET c.name = '{name}',
                c.court = '{court}',
                c.year = {year},
                c.bench_size = {bench_size},
                c.bench_type = '{bench_type}',
                c.matter_type = '{matter_type}',
                c.summary = '{summary}'
        """)
        self.stats["cases"] += 1
        return citation

    def upsert_legal_issue(self, case_citation: str, issue: dict):
        text = escape(issue.get("text", ""))
        issue_type = escape(issue.get("issue_type", "substantive"))
        if not text:
            return

        self.run(f"""
            MATCH (c:Case {{citation: '{case_citation}'}})
            MERGE (i:LegalIssue {{text: '{text}'}})
            SET i.issue_type = '{issue_type}'
            MERGE (c)-[:DECIDES]->(i)
        """)
        self.stats["issues"] += 1

    def upsert_rule(self, case_citation: str, rule: dict):
        text = escape(rule.get("text", ""))
        source = escape(rule.get("source", "precedent"))
        if not text:
            return

        self.run(f"""
            MATCH (c:Case {{citation: '{case_citation}'}})
            MERGE (r:Rule {{text: '{text}'}})
            SET r.source = '{source}'
            MERGE (c)-[:APPLIES_RULE]->(r)
        """)
        self.stats["rules"] += 1

    def upsert_statute_ref(self, case_citation: str, statute_ref: dict):
        statute_name = escape(statute_ref.get("statute_name", ""))
        section_num = escape(statute_ref.get("section_number", ""))
        purpose = escape(statute_ref.get("purpose", "relied_upon"))
        if not statute_name:
            return

        # Ensure Statute node exists
        self.run(f"""
            MERGE (s:Statute {{name: '{statute_name}'}})
        """)
        self.stats["statutes"] += 1

        if section_num:
            # Ensure Section node exists and link to Statute
            self.run(f"""
                MERGE (sec:Section {{number: '{section_num}', statute: '{statute_name}'}})
                WITH sec
                MATCH (s:Statute {{name: '{statute_name}'}})
                MERGE (sec)-[:PART_OF]->(s)
            """)
            self.stats["sections"] += 1

            # Link Case to Section
            self.run(f"""
                MATCH (c:Case {{citation: '{case_citation}'}})
                MATCH (sec:Section {{number: '{section_num}', statute: '{statute_name}'}})
                MERGE (c)-[:CITES_STATUTE {{purpose: '{purpose}'}}]->(sec)
            """)
        else:
            # Link Case directly to Statute if no section
            self.run(f"""
                MATCH (c:Case {{citation: '{case_citation}'}})
                MATCH (s:Statute {{name: '{statute_name}'}})
                MERGE (c)-[:CITES_STATUTE {{purpose: '{purpose}'}}]->(s)
            """)

    def upsert_precedent_ref(self, source_citation: str, precedent: dict):
        target_citation = escape(precedent.get("citation", ""))
        target_name = escape(precedent.get("case_name", ""))
        relationship = precedent.get("relationship", "CITES")
        proposition = escape(precedent.get("proposition", ""))
        conflict_type = escape(precedent.get("conflict_type", ""))

        if not target_citation and not target_name:
            return

        # Ensure target Case node exists (stub if not yet ingested)
        identifier = target_citation or target_name
        self.run(f"""
            MERGE (t:Case {{citation: '{escape(identifier)}'}})
            SET t.name = CASE WHEN t.name IS NULL OR t.name = '' 
                         THEN '{target_name}' ELSE t.name END
        """)

        # Validate relationship type
        valid_rels = {"CITES", "OVERRULES", "DISTINGUISHES", "CONFLICTS_WITH", "NARROWED_BY"}
        if relationship not in valid_rels:
            relationship = "CITES"

        if relationship == "CONFLICTS_WITH" and conflict_type:
            self.run(f"""
                MATCH (src:Case {{citation: '{source_citation}'}})
                MATCH (tgt:Case {{citation: '{escape(identifier)}'}})
                MERGE (src)-[r:CONFLICTS_WITH]->(tgt)
                SET r.conflict_type = '{conflict_type}',
                    r.proposition = '{proposition}',
                    r.unresolved = true
            """)
        else:
            self.run(f"""
                MATCH (src:Case {{citation: '{source_citation}'}})
                MATCH (tgt:Case {{citation: '{escape(identifier)}'}})
                MERGE (src)-[r:{relationship}]->(tgt)
                SET r.proposition = '{proposition}'
            """)

        self.stats["precedent_rels"] += 1

    def upsert_procedural_events(self, case_citation: str, events: list[str]):
        """Create a chain of ProceduralEvent nodes linked by PRECEDES."""
        if not events:
            return

        prev_event_id = None
        for i, event_type in enumerate(events):
            event_type_clean = escape(event_type.strip())
            event_id = f"{case_citation}_{event_type_clean}_{i}"
            event_id_clean = escape(event_id)

            self.run(f"""
                MERGE (e:ProceduralEvent {{event_id: '{event_id_clean}'}})
                SET e.event_type = '{event_type_clean}'
                WITH e
                MATCH (c:Case {{citation: '{case_citation}'}})
                MERGE (c)-[:INVOLVES_EVENT]->(e)
            """)
            self.stats["procedural_events"] += 1

            if prev_event_id:
                self.run(f"""
                    MATCH (prev:ProceduralEvent {{event_id: '{escape(prev_event_id)}'}})
                    MATCH (curr:ProceduralEvent {{event_id: '{event_id_clean}'}})
                    MERGE (prev)-[:PRECEDES]->(curr)
                """)

            prev_event_id = event_id

    def upsert_outcome(self, case_citation: str, conclusion: str, outcome_type: str):
        conclusion_clean = escape(conclusion)
        outcome_type_clean = escape(outcome_type)
        if not conclusion_clean:
            return

        self.run(f"""
            MATCH (c:Case {{citation: '{case_citation}'}})
            MERGE (o:Outcome {{text: '{conclusion_clean}'}})
            SET o.outcome_type = '{outcome_type_clean}'
            MERGE (c)-[:RESULTS_IN]->(o)
        """)

    # ----------------------------------------------------------------
    # Full IRAC document loader
    # ----------------------------------------------------------------

    def load_irac(self, irac: dict):
        """Load a complete IRAC document into the graph."""
        case_data = irac.get("case", {})
        case_citation = self.upsert_case(case_data)

        for issue in irac.get("issues", []):
            self.upsert_legal_issue(case_citation, issue)

        for rule in irac.get("rules", []):
            self.upsert_rule(case_citation, rule)

        for statute_ref in irac.get("statutes_cited", []):
            self.upsert_statute_ref(case_citation, statute_ref)

        for precedent in irac.get("precedents_cited", []):
            self.upsert_precedent_ref(case_citation, precedent)

        self.upsert_procedural_events(
            case_citation,
            irac.get("procedural_events", [])
        )

        self.upsert_outcome(
            case_citation,
            irac.get("conclusion", ""),
            irac.get("outcome_type", "dismissed")
        )

        return case_citation


def clear_graph(graph):
    console.print("[yellow]Clearing graph...[/yellow]")
    graph.query("MATCH (n) DETACH DELETE n")
    console.print("[yellow]Graph cleared.[/yellow]")


def main():
    parser = argparse.ArgumentParser(
        description="Load IRAC JSON files into FalkorDB"
    )
    parser.add_argument("--input", required=True,
                        help="Path to IRAC JSON file or directory")
    parser.add_argument("--clear", action="store_true",
                        help="Clear the graph before loading (use with caution)")
    args = parser.parse_args()

    try:
        graph = get_graph()
        graph.query("RETURN 1")
        console.print("[green]Connected to FalkorDB[/green]")
    except Exception as e:
        console.print(f"[red]Cannot connect to FalkorDB: {e}[/red]")
        return

    if args.clear:
        clear_graph(graph)

    loader = GraphLoader(graph)

    input_path = Path(args.input)
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(input_path.glob("**/*_irac.json"))
        console.print(f"Found [bold]{len(files)}[/bold] IRAC JSON files")
    else:
        console.print(f"[red]Input path not found: {input_path}[/red]")
        return

    loaded = []
    failed = []

    for f in tqdm(files, desc="Loading to FalkorDB"):
        try:
            with open(f, encoding="utf-8") as jf:
                irac = json.load(jf)
            citation = loader.load_irac(irac)
            loaded.append(citation)
        except Exception as e:
            failed.append((f.name, str(e)))
            console.print(f"[red]Failed:[/red] {f.name}: {e}")

    console.print(f"\n[green]Loaded {len(loaded)} judgments.[/green]")
    if failed:
        console.print(f"[red]{len(failed)} failed.[/red]")

    console.print("\nGraph statistics:")
    for key, val in loader.stats.items():
        console.print(f"  {key}: {val}")

    console.print("\nNext step: python agents/retrieval_agent.py --query 'your query here'")


if __name__ == "__main__":
    main()
