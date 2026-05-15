# falkor-irac

**Graph-Constrained Legal Reasoning for Indian Judicial AI**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FalkorDB](https://img.shields.io/badge/graph-FalkorDB-orange.svg)](https://www.falkordb.com/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.14665-red.svg)](https://arxiv.org/abs/2605.14665)
[![Dataset](https://img.shields.io/badge/HuggingFace-inIRAC-yellow.svg)](https://huggingface.co/datasets/joyboseroy/inIRAC)

> *"Courts are effectively graph traversal engines disguised as prose."*

---

## Overview

`falkor-irac` is a **verified graph reasoning framework** for Indian legal AI. It replaces vector-only RAG with structured legal cognition: every LLM-generated answer must be grounded to a **valid reasoning path** in a knowledge graph before it is returned to the user.

The system is built around three ideas:

1. **IRAC Graph Schema** — Supreme Court and High Court judgments are ingested not as text chunks but as structured IRAC (Issue, Rule, Analysis, Conclusion) nodes, enriched with procedural state transitions and precedent relationships.

2. **Graph-Constrained Generation** — The LLM proposes an answer; a Verifier Agent checks whether a supporting path exists in the FalkorDB graph. If no valid path exists, the answer is rejected or flagged, not returned.

3. **Conflict-Aware Reasoning** — The system detects doctrinal conflicts (coordinate bench disagreements, per incuriam citations, distinguished precedents) as a first-class output rather than silently preferring one path.

This is **not** a legal chatbot. It is a **reasoning substrate** — reusable infrastructure that can later power compliance AI, telecom regulation analysis, patent drafting, and any domain requiring verifiable multi-step justification.

**arXiv paper:** https://arxiv.org/abs/2605.14665  
**Dataset:** https://huggingface.co/datasets/joyboseroy/inIRAC

---

## Why Not Just RAG?

Standard retrieval-augmented generation treats legal reasoning as nearest-neighbour search over text embeddings. This fails in law for three structural reasons:

| RAG Assumption | Legal Reality |
|---|---|
| Semantic similarity ≈ relevance | A landmark precedent may use completely different vocabulary from your query |
| Retrieve → Generate → Done | Legal reasoning requires: retrieve → check procedural validity → verify statute freshness → detect conflicts → generate |
| Hallucination is a quality issue | In law, a hallucinated precedent is a professional liability |
| One best answer exists | Courts regularly hold conflicting positions across coordinate benches |

The core insight: **legal reasoning is constrained symbolic traversal over a precedent graph, not fuzzy similarity search**.

---

## Architecture

```
User Query
    │
    ▼
Retrieval Agent
(graph traversal + precedent candidate generation)
    │
    ▼
IRAC Graph (FalkorDB)
(candidate reasoning paths)
    │
    ▼
LLM Synthesis
(path-guided answer generation)
    │
    ▼
Verifier Agent  ◄──── falsifiability oracle
(does a valid citation path exist?)
    │
    ├── YES → Path-backed Answer + citation chain
    └── NO  → Conflict flagged / answer rejected
```

The Verifier is not a generative agent. It is a **falsifiability oracle**: given a proposed reasoning chain, it checks for path existence in the graph. Binary output. No generation. This is the anti-hallucination mechanism.

---

## Graph Schema

### Node Types

| Node | Description |
|---|---|
| `CASE` | Judgment (court, bench, year, citation) |
| `JUDGE` | Author or bench member |
| `STATUTE` | Act or ordinance |
| `SECTION` | Specific provision |
| `LEGAL_ISSUE` | The question before the court |
| `RULE` | Legal principle extracted from judgment |
| `ARGUMENT` | Petitioner or respondent position |
| `PRECEDENT` | Prior case cited in support |
| `PROCEDURAL_EVENT` | Bail, hearing, appeal, delay, interim relief |
| `OUTCOME` | Conclusion of the court |
| `JURISDICTION` | Court and territorial scope |

### Relationship Types

| Relationship | Meaning |
|---|---|
| `CITES` | Case relies on precedent |
| `OVERRULES` | Later judgment expressly overrules earlier |
| `DISTINGUISHES` | Limits earlier holding on facts |
| `CONFLICTS_WITH` | Coordinate bench disagreement (with `conflict_type` attribute) |
| `RESOLVED_BY` | Points to later resolution (larger bench / full bench) |
| `APPLIES_RULE` | Case applies a statutory rule to facts |
| `SUPPORTS_ARGUMENT` | Precedent supports a party's argument |
| `TRIGGERS` | Procedural event triggers subsequent event |
| `PRECEDES` | Temporal ordering of procedural events |
| `RESULTS_IN` | Facts or procedure results in outcome |
| `NARROWED_BY` | Prior holding narrowed by later case on facts |

The procedural layer (`TRIGGERS`, `PRECEDES`) is what distinguishes this schema from citation-network-only approaches. It enables reasoning over **timelines and litigation flow**, not just doctrinal trees.

---

## Conflict Detection

When the Verifier finds multiple valid paths supporting contradictory conclusions, it returns:

```python
{
    "answer": "...",
    "supporting_paths": [path_A, path_B],
    "conflict": True,
    "conflict_type": "coordinate_bench",   # or: overruled | per_incuriam | distinguished
    "resolution": "unresolved — flag for human review",
    "confidence": "low"
}
```

The system **never silently prefers one path**. Doctrinal conflict is a first-class output.

---

## Datasets

| Dataset | Use |
|---|---|
| [ILDC](https://github.com/Law-AI/ILDC) | Supreme Court judgments (ACL 2021) |
| [NyayaAnumana](https://github.com/Law-AI/NyayaAnumana) | Largest Indian legal judgment prediction dataset |
| [MILPaC](https://github.com/Law-AI/MILPaC) | Multilingual Indian legal parallel corpus (9 Indic languages) |
| [IndicLegalQA](https://huggingface.co/) | Legal QA in Indian judicial context (2025) |
| Indian Bail Orders Dataset | 20+ structured attributes per case — bail, IPC sections, judgment reason |

---

## Repository Structure

```
falkor-irac/
├── data/
│   ├── raw/               # Downloaded judgment PDFs
│   └── processed/         # Extracted IRAC JSON
├── graph_schema/
│   ├── schema.cypher      # FalkorDB schema definition
│   └── sample_graph.json  # Small example for testing
├── ingestion/
│   ├── pdf_extractor.py   # PDF → structured text
│   ├── irac_parser.py     # LLM-assisted IRAC extraction
│   └── graph_loader.py    # Populates FalkorDB
├── agents/
│   ├── retrieval_agent.py   # Graph traversal + precedent candidates
│   ├── constraint_agent.py  # Statute consistency checking
│   ├── temporal_agent.py    # Procedural validity across timelines
│   └── verifier_agent.py   # Citation path validation (falsifiability oracle)
├── evaluation/
│   ├── path_validity.py     # Citation grounding accuracy
│   ├── procedural_consistency.py
│   └── hallucination_rate.py  # Hallucinated precedent detection
├── notebooks/
│   ├── 01_ingest_judgment.ipynb
│   ├── 02_explore_graph.ipynb
│   └── 03_query_with_verification.ipynb
├── requirements.txt
└── README.md
```

---

## Quickstart (v0.1)

### Prerequisites

```bash
# FalkorDB (Docker)
docker run -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest

# Python dependencies
pip install -r requirements.txt
```

### Ingest a judgment

```bash
python ingestion/pdf_extractor.py --input data/raw/sc_judgment.pdf --output data/processed/

python ingestion/irac_parser.py --input data/processed/sc_judgment.json

python ingestion/graph_loader.py --input data/processed/sc_judgment_irac.json
```

### Query with graph-grounded verification

```python
from agents.retrieval_agent import RetrievalAgent
from agents.verifier_agent import VerifierAgent

retriever = RetrievalAgent(graph_url="redis://localhost:6379")
verifier = VerifierAgent(requires_citation_path=True)

result = retriever.query("What precedents apply to bail denial on non-appearance?")
verified = verifier.check(result)

print(verified["answer"])
print(verified["citation_path"])
print(verified["conflict"])  # True if conflicting precedents found
```

---

## Evaluation Metrics

Standard BLEU/ROUGE scores are insufficient for legal reasoning. This repo evaluates on:

| Metric | What it measures |
|---|---|
| **Citation Grounding Accuracy** | Does every claim map to a real node/path in the graph? |
| **Path Validity Rate** | % of answers with a valid supporting graph path |
| **Procedural Consistency** | Are procedural sequences temporally valid? |
| **Statute Freshness** | Are cited statutes still in force? |
| **Hallucinated Precedent Rate** | % of cited cases that do not exist in the graph |
| **Conflict Detection Rate** | % of genuine doctrinal conflicts correctly flagged |

---

## Roadmap

- **v0.1** — Ingest SC judgment → extract IRAC → populate FalkorDB → answer with citation path
- **v0.2** — Procedural state transitions + timeline reasoning
- **v0.3** — Conflict detection with `CONFLICTS_WITH` / `RESOLVED_BY`
- **v0.4** — Indic language layer via Bhashini API (Hindi, Bengali, Tamil)
- **v0.5** — Evaluation suite with path validity benchmarks
- **v1.0** — Full pipeline on ILDC + NyayaAnumana datasets

---

## Related Work

- Song et al. (2026). *Knowledge Graph-Assisted LLM Post-Training for Enhanced Legal Reasoning*. arXiv:2601.13806
- Han (2026). *Trustworthy Legal Reasoning: A Comprehensive Survey*. Preprints.org
- Karna (2026). *A Hybrid RAG-LLaMA Framework for Scalable and Accurate Interpretation of Legal Texts*. Applied Artificial Intelligence, 40(1)
- Malik et al. (2021). *ILDC for CJPE: Indian Legal Documents Corpus for Court Judgment Prediction and Explanation*. ACL 2021
- Awasekar (2026). *NyayaSakhi–SWATI: India's First Statute-Aligned, Retrieval-Augmented Legal AI*. JEET

---

## Related Projects

- [NyayaSaar-LoRA](https://github.com/joyboseroy/NyayaSaar-LoRA)  
  PEFT/QLoRA-based simplification of structured Indian legal reasoning into plain English.

- [inIRAC Dataset](https://huggingface.co/datasets/joyboseroy/inIRAC)  
  Structured Indian legal reasoning dataset in IRAC format.

---

## Citation

If you use this work, please cite:

```bibtex
@software{falkor_irac_2026,
  title  = {falkor-irac: Graph-Constrained Legal Reasoning for Indian Judicial AI},
  author = {Bose, Joy},
  year   = {2026},
  url    = {https://github.com/joybose/falkor-irac}
}
```

---

## Contributing

Contributions welcome — especially:
- Additional judgment ingestion pipelines
- Indic language support
- Evaluation dataset curation
- FalkorDB schema refinements

Please open an issue before submitting a large PR.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*This project is part of ongoing research into verified graph reasoning for Indian legal AI. A companion arXiv paper is in preparation.*
