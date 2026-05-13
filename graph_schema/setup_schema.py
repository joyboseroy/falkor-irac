"""
graph_schema/setup_schema.py

Connects to FalkorDB and sets up indexes for the legal knowledge graph.
Run once before ingesting any judgments.

Usage:
    python graph_schema/setup_schema.py
"""

import os
from dotenv import load_dotenv
from falkordb import FalkorDB
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()


def get_graph():
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", 6381))
    graph_name = os.getenv("FALKORDB_GRAPH_NAME", "legal_graph")

    db = FalkorDB(host=host, port=port)
    return db.select_graph(graph_name)


INDEX_QUERIES = [
    ("Case citation", "CREATE INDEX ON :Case(citation)"),
    ("Case year",     "CREATE INDEX ON :Case(year)"),
    ("Case court",    "CREATE INDEX ON :Case(court)"),
    ("Case matter",   "CREATE INDEX ON :Case(matter_type)"),
    ("Statute name",  "CREATE INDEX ON :Statute(name)"),
    ("Section num",   "CREATE INDEX ON :Section(number)"),
    ("Judge name",    "CREATE INDEX ON :Judge(name)"),
    ("ProcEvent type","CREATE INDEX ON :ProceduralEvent(event_type)"),
]


def setup_indexes(graph):
    console.print("\n[bold]Creating indexes...[/bold]")
    for label, query in INDEX_QUERIES:
        try:
            graph.query(query)
            console.print(f"  [green]OK[/green]  {label}")
        except Exception as e:
            # Index may already exist
            if "already indexed" in str(e).lower():
                console.print(f"  [yellow]EXISTS[/yellow]  {label}")
            else:
                console.print(f"  [red]FAIL[/red]  {label}: {e}")


def verify_connection(graph):
    result = graph.query("RETURN 'connected' AS status")
    return result.result_set[0][0] == "connected"


def main():
    console.print(Panel.fit(
        "[bold blue]falkor-irac[/bold blue]: Schema Setup",
        subtitle="FalkorDB Legal Knowledge Graph"
    ))

    try:
        graph = get_graph()
        if verify_connection(graph):
            console.print("[green]Connected to FalkorDB[/green]")
        else:
            console.print("[red]Connection check failed[/red]")
            return
    except Exception as e:
        console.print(f"[red]Cannot connect to FalkorDB: {e}[/red]")
        console.print("\nMake sure FalkorDB is running:")
        console.print("  docker run -p 6381:6379 -p 3000:3000 falkordb/falkordb:latest")
        return

    setup_indexes(graph)

    console.print("\n[green]Schema setup complete.[/green]")
    console.print("Next step: python ingestion/graph_loader.py --help\n")


if __name__ == "__main__":
    main()
