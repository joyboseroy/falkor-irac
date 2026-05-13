"""
ingestion/load_sample_graph.py

Loads the sample graph (graph_schema/sample_graph.json) into FalkorDB.
Use this to get a working graph quickly without needing real judgment PDFs.

Usage:
    python ingestion/load_sample_graph.py
    python ingestion/load_sample_graph.py --clear
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from falkordb import FalkorDB
from rich.console import Console

load_dotenv()
console = Console()

SAMPLE_GRAPH_PATH = Path(__file__).parent.parent / "graph_schema" / "sample_graph.json"


def get_graph():
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", 6381))
    graph_name = os.getenv("FALKORDB_GRAPH_NAME", "legal_graph")
    db = FalkorDB(host=host, port=port)
    return db.select_graph(graph_name)


def escape(s) -> str:
    if s is None:
        return ""
    return str(s).replace("\\", "\\\\").replace("'", "\\'")


def load_sample(graph, data: dict):
    q = graph.query

    # Cases
    for case in data.get("cases", []):
        q(f"""
            MERGE (c:Case {{citation: '{escape(case['citation'])}'}})
            SET c.name = '{escape(case['name'])}',
                c.court = '{escape(case['court'])}',
                c.year = {case['year']},
                c.bench_size = {case.get('bench_size', 2)},
                c.bench_type = '{escape(case.get('bench_type', 'division'))}',
                c.matter_type = '{escape(case.get('matter_type', 'general'))}',
                c.summary = '{escape(case.get('summary', ''))}'
        """)

    # Statutes
    for statute in data.get("statutes", []):
        q(f"""
            MERGE (s:Statute {{name: '{escape(statute['name'])}'}})
            SET s.short_name = '{escape(statute.get('short_name', ''))}',
                s.year = {statute.get('year', 0)},
                s.repealed = {str(statute.get('repealed', False)).lower()}
        """)

    # Sections
    for section in data.get("sections", []):
        q(f"""
            MERGE (sec:Section {{number: '{escape(section['number'])}',
                                 statute: '{escape(section['statute'])}'}})
            SET sec.title = '{escape(section.get('title', ''))}',
                sec.repealed = {str(section.get('repealed', False)).lower()}
            WITH sec
            MATCH (s:Statute {{name: '{escape(section['statute'])}'}})
            MERGE (sec)-[:PART_OF]->(s)
        """)

    # Legal issues
    for issue in data.get("issues", []):
        q(f"""
            MATCH (c:Case {{citation: '{escape(issue['case_citation'])}'}})
            MERGE (i:LegalIssue {{text: '{escape(issue['text'])}'}})
            SET i.issue_type = '{escape(issue.get('issue_type', 'substantive'))}'
            MERGE (c)-[:DECIDES]->(i)
        """)

    # Precedent relationships
    for rel in data.get("precedent_relationships", []):
        rel_type = rel.get("relationship", "CITES")
        q(f"""
            MATCH (src:Case {{citation: '{escape(rel['source'])}'}})
            MATCH (tgt:Case {{citation: '{escape(rel['target'])}'}})
            MERGE (src)-[r:{rel_type}]->(tgt)
            SET r.proposition = '{escape(rel.get('proposition', ''))}'
        """)

    # Statute citations
    for sc in data.get("statute_citations", []):
        q(f"""
            MATCH (c:Case {{citation: '{escape(sc['case_citation'])}'}})
            MATCH (sec:Section {{number: '{escape(sc['section_number'])}',
                                  statute: '{escape(sc['statute_name'])}'}})
            MERGE (c)-[r:CITES_STATUTE]->(sec)
            SET r.purpose = '{escape(sc.get('purpose', 'relied_upon'))}'
        """)

    # Procedural event chains
    for seq in data.get("procedural_sequences", []):
        case_citation = escape(seq["case_citation"])
        prev_id = None
        for i, event_type in enumerate(seq["events"]):
            event_id = f"{seq['case_citation']}_{event_type}_{i}"
            eid = escape(event_id)
            etype = escape(event_type)
            q(f"""
                MERGE (e:ProceduralEvent {{event_id: '{eid}'}})
                SET e.event_type = '{etype}'
                WITH e
                MATCH (c:Case {{citation: '{case_citation}'}})
                MERGE (c)-[:INVOLVES_EVENT]->(e)
            """)
            if prev_id:
                q(f"""
                    MATCH (prev:ProceduralEvent {{event_id: '{escape(prev_id)}'}})
                    MATCH (curr:ProceduralEvent {{event_id: '{eid}'}})
                    MERGE (prev)-[:PRECEDES]->(curr)
                """)
            prev_id = event_id


def verify_load(graph):
    result = graph.query("MATCH (c:Case) RETURN count(c) AS n")
    n_cases = result.result_set[0][0]
    result = graph.query("MATCH (s:Statute) RETURN count(s) AS n")
    n_statutes = result.result_set[0][0]
    result = graph.query("MATCH ()-[r]->() RETURN count(r) AS n")
    n_rels = result.result_set[0][0]
    return n_cases, n_statutes, n_rels


def main():
    parser = argparse.ArgumentParser(description="Load sample graph into FalkorDB")
    parser.add_argument("--clear", action="store_true", help="Clear graph first")
    args = parser.parse_args()

    try:
        graph = get_graph()
        graph.query("RETURN 1")
        console.print("[green]Connected to FalkorDB[/green]")
    except Exception as e:
        console.print(f"[red]Cannot connect to FalkorDB: {e}[/red]")
        console.print("\nStart FalkorDB with:")
        console.print("  docker run -p 6381:6379 -p 3000:3000 falkordb/falkordb:latest")
        return

    if args.clear:
        graph.query("MATCH (n) DETACH DELETE n")
        console.print("[yellow]Graph cleared.[/yellow]")

    with open(SAMPLE_GRAPH_PATH, encoding="utf-8") as f:
        data = json.load(f)

    console.print(f"Loading sample graph from {SAMPLE_GRAPH_PATH.name}...")
    load_sample(graph, data)

    n_cases, n_statutes, n_rels = verify_load(graph)
    console.print(f"\n[green]Sample graph loaded.[/green]")
    console.print(f"  Cases: {n_cases}")
    console.print(f"  Statutes: {n_statutes}")
    console.print(f"  Relationships: {n_rels}")

    console.print("\nTry a query:")
    console.print("  python agents/orchestrator.py --query 'What are the grounds for bail denial?' --matter-type bail")


if __name__ == "__main__":
    main()
