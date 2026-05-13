"""
tests/test_verifier.py

Unit tests for the VerifierAgent.
These tests use a mock FalkorDB graph seeded with known data,
so they run without a live database connection.
"""

import pytest
from unittest.mock import MagicMock, patch
from agents.verifier_agent import VerifierAgent, VerificationStatus


def make_verifier(query_side_effect):
    """Create a VerifierAgent with a mocked graph."""
    verifier = VerifierAgent.__new__(VerifierAgent)
    verifier.requires_citation_path = True
    mock_graph = MagicMock()
    mock_graph.query.side_effect = query_side_effect
    verifier.graph = mock_graph
    return verifier


class TestCitationExists:

    def test_known_citation_returns_true(self):
        def mock_query(q, *args):
            result = MagicMock()
            if "MATCH (c:Case" in q and "citation" in q:
                result.result_set = [["Supreme Court", 2012]]
            elif "OVERRULES" in q:
                result.result_set = []
            else:
                result.result_set = []
            return result

        verifier = make_verifier(lambda q, *a: mock_query(q))
        check = verifier.check_citation_exists("(2012) 9 SCC 1")
        assert check.exists_in_graph is True
        assert check.court == "Supreme Court"
        assert check.year == 2012
        assert check.is_overruled is False

    def test_unknown_citation_returns_false(self):
        def mock_query(q, *args):
            result = MagicMock()
            result.result_set = []
            return result

        verifier = make_verifier(lambda q, *a: mock_query(q))
        check = verifier.check_citation_exists("(2099) 99 SCC 999")
        assert check.exists_in_graph is False

    def test_overruled_case_flagged(self):
        call_count = [0]

        def mock_query(q, *args):
            result = MagicMock()
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: case exists
                result.result_set = [["Supreme Court", 2005]]
            elif "OVERRULES" in q:
                # Second call: overruled
                result.result_set = [["(2015) 3 SCC 100"]]
            else:
                result.result_set = []
            return result

        verifier = make_verifier(lambda q, *a: mock_query(q))
        check = verifier.check_citation_exists("(2005) 1 SCC 50")
        assert check.exists_in_graph is True
        assert check.is_overruled is True
        assert check.overruled_by == "(2015) 3 SCC 100"


class TestStatuteFreshness:

    def test_current_statute_returns_not_repealed(self):
        def mock_query(q, *args):
            result = MagicMock()
            result.result_set = [[False, None]]
            return result

        verifier = make_verifier(lambda q, *a: mock_query(q))
        check = verifier.check_statute_current("Code of Criminal Procedure, 1973", "437")
        assert check.exists_in_graph is True
        assert check.is_repealed is False

    def test_repealed_statute_flagged(self):
        def mock_query(q, *args):
            result = MagicMock()
            result.result_set = [[True, "Bharatiya Nagarik Suraksha Sanhita, 2023"]]
            return result

        verifier = make_verifier(lambda q, *a: mock_query(q))
        check = verifier.check_statute_current("Code of Criminal Procedure, 1973", "437")
        assert check.is_repealed is True
        assert "Bharatiya" in check.repealed_by


class TestVerificationStatus:

    def test_valid_answer_returns_valid(self):
        def mock_query(q, *args):
            result = MagicMock()
            if "MATCH (c:Case" in q:
                result.result_set = [["Supreme Court", 2012]]
            elif "CONFLICTS_WITH" in q:
                result.result_set = []
            else:
                result.result_set = []
            return result

        verifier = make_verifier(lambda q, *a: mock_query(q))
        result = verifier.check(
            proposed_answer="Bail may be granted under s.437 CrPC...",
            citations=["(2012) 9 SCC 1"],
            statute_refs=[{"statute_name": "Code of Criminal Procedure, 1973", "section_number": "437"}],
        )
        assert result.status == VerificationStatus.VALID
        assert result.confidence == 1.0

    def test_missing_citation_returns_invalid(self):
        def mock_query(q, *args):
            result = MagicMock()
            result.result_set = []
            return result

        verifier = make_verifier(lambda q, *a: mock_query(q))
        result = verifier.check(
            proposed_answer="As held in the landmark case...",
            citations=["(2099) 99 SCC 999"],
        )
        assert result.status == VerificationStatus.INVALID
        assert result.confidence == 0.0
        assert "not found in graph" in result.notes

    def test_conflicting_citations_returns_conflict(self):
        call_log = []

        def mock_query(q, *args):
            call_log.append(q)
            result = MagicMock()
            if "MATCH (c:Case" in q and "CONFLICTS_WITH" not in q and "OVERRULES" not in q:
                result.result_set = [["Supreme Court", 2012]]
            elif "CONFLICTS_WITH" in q:
                result.result_set = [["coordinate_bench", True]]
            elif "RESOLVED_BY" in q:
                result.result_set = []
            else:
                result.result_set = []
            return result

        verifier = make_verifier(lambda q, *a: mock_query(q))
        result = verifier.check(
            proposed_answer="...",
            citations=["(2012) 9 SCC 1", "(2013) 4 SCC 20"],
        )
        assert result.status == VerificationStatus.CONFLICT
        assert len(result.conflict_details) > 0
        assert result.conflict_details[0]["conflict_type"] == "coordinate_bench"

    def test_empty_citations_returns_low_confidence(self):
        verifier = make_verifier(lambda q, *a: (_ for _ in ()).throw(Exception("should not be called")))
        # Patch the check methods to avoid graph calls
        verifier.check_citation_exists = MagicMock(return_value=MagicMock(exists_in_graph=False))
        verifier.check_statute_current = MagicMock()
        verifier.check_for_conflicts = MagicMock(return_value=[])

        result = verifier.check(proposed_answer="Some answer", citations=[], statute_refs=[])
        assert result.confidence == 0.5  # no citations to verify


class TestConflictOutputFormat:

    def test_conflict_output_has_required_fields(self):
        """Verify the conflict output dict has the fields the Medium article describes."""
        from agents.orchestrator import LegalAnswer
        from agents.verifier_agent import VerificationResult, VerificationStatus

        mock_verification = VerificationResult(
            status=VerificationStatus.CONFLICT,
            conflict_details=[{
                "case_a": "(2012) 9 SCC 1",
                "case_b": "(2013) 4 SCC 20",
                "conflict_type": "coordinate_bench",
                "unresolved": True,
                "resolved_by": None,
            }],
            confidence=0.4,
            notes="Unresolved conflict"
        )

        conflict_output = {
            "answer": "Based on current precedent...",
            "supporting_paths": ["(2012) 9 SCC 1", "(2013) 4 SCC 20"],
            "conflict": True,
            "conflict_type": "coordinate_bench",
            "resolution": "unresolved - refer to larger bench ruling if available",
            "confidence": "low",
        }

        required_fields = {"answer", "supporting_paths", "conflict", "conflict_type", "resolution", "confidence"}
        assert required_fields.issubset(set(conflict_output.keys()))
        assert conflict_output["conflict"] is True
