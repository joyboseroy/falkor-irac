"""
evaluation/metrics.py

Graph-native evaluation metrics for legal reasoning.

Metrics:
    - citation_grounding_accuracy   : % of cited cases that exist in the graph
    - path_validity_rate            : % of answers with a valid supporting citation path
    - hallucinated_precedent_rate   : % of cited cases NOT in the graph
    - procedural_consistency        : % of procedural event sequences that are temporally valid
    - conflict_detection_rate       : % of genuine conflicts correctly flagged
    - false_conflict_rate           : % of non-conflicting answers incorrectly flagged
    - statute_freshness_rate        : % of cited statutes that are current (not repealed)

These replace BLEU/ROUGE for legal reasoning evaluation.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from agents.verifier_agent import VerifierAgent, VerificationStatus
from agents.retrieval_agent import RetrievalAgent


@dataclass
class EvaluationSample:
    """A single evaluation example."""
    query: str
    answer: str
    citations: list[str]
    statute_refs: list[dict] = field(default_factory=list)
    # Ground truth labels (from human annotation)
    ground_truth_citations: list[str] = field(default_factory=list)
    has_genuine_conflict: bool = False        # annotated ground truth
    procedural_events: list[str] = field(default_factory=list)


@dataclass
class MetricResult:
    name: str
    score: float
    numerator: int
    denominator: int
    details: list[dict] = field(default_factory=list)

    def __str__(self):
        return (
            f"{self.name}: {self.score:.3f} "
            f"({self.numerator}/{self.denominator})"
        )


class LegalReasoningEvaluator:
    """
    Evaluates a set of LegalAnswer objects against graph-native metrics.
    """

    def __init__(self):
        self.verifier = VerifierAgent(requires_citation_path=False)
        self.retriever = RetrievalAgent()

    def citation_grounding_accuracy(
        self, samples: list[EvaluationSample]
    ) -> MetricResult:
        """
        For each answer, what fraction of cited cases actually exist in the graph?
        """
        total_citations = 0
        grounded_citations = 0
        details = []

        for sample in samples:
            for citation in sample.citations:
                check = self.verifier.check_citation_exists(citation)
                total_citations += 1
                if check.exists_in_graph:
                    grounded_citations += 1
                else:
                    details.append({
                        "query": sample.query[:80],
                        "missing_citation": citation
                    })

        score = grounded_citations / total_citations if total_citations else 0.0
        return MetricResult(
            name="citation_grounding_accuracy",
            score=score,
            numerator=grounded_citations,
            denominator=total_citations,
            details=details,
        )

    def hallucinated_precedent_rate(
        self, samples: list[EvaluationSample]
    ) -> MetricResult:
        """
        What fraction of cited cases are NOT in the graph?
        Lower is better.
        """
        result = self.citation_grounding_accuracy(samples)
        hallucination_score = 1.0 - result.score
        return MetricResult(
            name="hallucinated_precedent_rate",
            score=hallucination_score,
            numerator=result.denominator - result.numerator,
            denominator=result.denominator,
            details=result.details,
        )

    def path_validity_rate(
        self, samples: list[EvaluationSample]
    ) -> MetricResult:
        """
        What fraction of answers have at least one valid citation path in the graph?
        An answer is path-valid if ALL its citations exist and none are overruled.
        """
        valid_answers = 0
        details = []

        for sample in samples:
            if not sample.citations:
                details.append({"query": sample.query[:80], "status": "no_citations"})
                continue

            verification = self.verifier.check(
                proposed_answer=sample.answer,
                citations=sample.citations,
                statute_refs=sample.statute_refs,
            )

            if verification.status == VerificationStatus.VALID:
                valid_answers += 1
            else:
                details.append({
                    "query": sample.query[:80],
                    "status": verification.status.value,
                    "notes": verification.notes,
                })

        score = valid_answers / len(samples) if samples else 0.0
        return MetricResult(
            name="path_validity_rate",
            score=score,
            numerator=valid_answers,
            denominator=len(samples),
            details=details,
        )

    def conflict_detection_rate(
        self, samples: list[EvaluationSample]
    ) -> MetricResult:
        """
        Among samples annotated as having genuine doctrinal conflicts,
        what fraction does the system correctly flag as CONFLICT?
        """
        conflicted_samples = [s for s in samples if s.has_genuine_conflict]
        correctly_flagged = 0
        details = []

        for sample in conflicted_samples:
            verification = self.verifier.check(
                proposed_answer=sample.answer,
                citations=sample.citations,
                statute_refs=sample.statute_refs,
            )
            if verification.status == VerificationStatus.CONFLICT:
                correctly_flagged += 1
            else:
                details.append({
                    "query": sample.query[:80],
                    "expected": "CONFLICT",
                    "got": verification.status.value,
                })

        score = correctly_flagged / len(conflicted_samples) if conflicted_samples else 0.0
        return MetricResult(
            name="conflict_detection_rate",
            score=score,
            numerator=correctly_flagged,
            denominator=len(conflicted_samples),
            details=details,
        )

    def false_conflict_rate(
        self, samples: list[EvaluationSample]
    ) -> MetricResult:
        """
        Among samples annotated as NOT having genuine conflicts,
        what fraction does the system incorrectly flag as CONFLICT?
        Lower is better.
        """
        non_conflicted_samples = [s for s in samples if not s.has_genuine_conflict]
        incorrectly_flagged = 0
        details = []

        for sample in non_conflicted_samples:
            verification = self.verifier.check(
                proposed_answer=sample.answer,
                citations=sample.citations,
                statute_refs=sample.statute_refs,
            )
            if verification.status == VerificationStatus.CONFLICT:
                incorrectly_flagged += 1
                details.append({
                    "query": sample.query[:80],
                    "false_conflict_citations": sample.citations,
                })

        score = incorrectly_flagged / len(non_conflicted_samples) if non_conflicted_samples else 0.0
        return MetricResult(
            name="false_conflict_rate",
            score=score,
            numerator=incorrectly_flagged,
            denominator=len(non_conflicted_samples),
            details=details,
        )

    def statute_freshness_rate(
        self, samples: list[EvaluationSample]
    ) -> MetricResult:
        """
        What fraction of cited statutes are current (not repealed)?
        """
        total = 0
        current = 0
        details = []

        for sample in samples:
            for ref in sample.statute_refs:
                check = self.verifier.check_statute_current(
                    ref.get("statute_name", ""),
                    ref.get("section_number", "")
                )
                total += 1
                if not check.is_repealed:
                    current += 1
                else:
                    details.append({
                        "query": sample.query[:80],
                        "stale_statute": ref.get("statute_name"),
                        "repealed_by": check.repealed_by,
                    })

        score = current / total if total else 1.0  # if no statutes cited, full score
        return MetricResult(
            name="statute_freshness_rate",
            score=score,
            numerator=current,
            denominator=total,
            details=details,
        )

    def run_all(self, samples: list[EvaluationSample]) -> dict[str, MetricResult]:
        """Run all metrics and return a summary dict."""
        return {
            "citation_grounding_accuracy": self.citation_grounding_accuracy(samples),
            "hallucinated_precedent_rate": self.hallucinated_precedent_rate(samples),
            "path_validity_rate": self.path_validity_rate(samples),
            "conflict_detection_rate": self.conflict_detection_rate(samples),
            "false_conflict_rate": self.false_conflict_rate(samples),
            "statute_freshness_rate": self.statute_freshness_rate(samples),
        }

    def to_dataframe(self, results: dict[str, MetricResult]) -> pd.DataFrame:
        rows = []
        for name, metric in results.items():
            rows.append({
                "metric": name,
                "score": round(metric.score, 4),
                "count": f"{metric.numerator}/{metric.denominator}",
            })
        return pd.DataFrame(rows)


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from rich.console import Console
    from rich.table import Table

    console = Console()
    parser = argparse.ArgumentParser(description="Run evaluation metrics")
    parser.add_argument("--samples", required=True,
                        help="Path to JSON file of evaluation samples")
    args = parser.parse_args()

    with open(args.samples, encoding="utf-8") as f:
        raw_samples = json.load(f)

    samples = [
        EvaluationSample(
            query=s["query"],
            answer=s["answer"],
            citations=s.get("citations", []),
            statute_refs=s.get("statute_refs", []),
            ground_truth_citations=s.get("ground_truth_citations", []),
            has_genuine_conflict=s.get("has_genuine_conflict", False),
        )
        for s in raw_samples
    ]

    console.print(f"\nEvaluating [bold]{len(samples)}[/bold] samples...\n")

    evaluator = LegalReasoningEvaluator()
    results = evaluator.run_all(samples)

    table = Table(title="falkor-irac Evaluation Results")
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Count", justify="right")

    for name, metric in results.items():
        colour = "green" if metric.score >= 0.8 else "yellow" if metric.score >= 0.5 else "red"
        # For hallucination and false conflict, lower is better
        if "hallucinated" in name or "false_conflict" in name:
            colour = "green" if metric.score <= 0.1 else "yellow" if metric.score <= 0.3 else "red"
        table.add_row(
            name,
            f"[{colour}]{metric.score:.3f}[/{colour}]",
            f"{metric.numerator}/{metric.denominator}"
        )

    console.print(table)
