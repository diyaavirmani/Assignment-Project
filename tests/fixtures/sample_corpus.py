"""
Fixture corpus for self-contained end-to-end eval smoke tests.

This module provides SAMPLE_CHUNKS: 10 hand-built chunks representing tiny
excerpts from 3GPP specifications (content abbreviated for test purposes) and
SAMPLE_QUERIES: 6 eval queries with known ground-truth, designed so that the
correct chunks are retrievable by any reasonable similarity function.

PURPOSE
-------
These fixtures exist to make the eval harness runnable end-to-end WITHOUT the
full 43,121-chunk index. They let CI and reviewers verify that every metric
function (Recall@k, MRR, nDCG@k, hit-rate, heuristic metrics) produces a
real, reproducible number against a controlled input.

IMPORTANT — these are a smoke-test fixture, not a quality benchmark.
  - 10 chunks vs 43,121 in the real index.
  - 6 queries vs 25 in the full golden set.
  - Numbers from this fixture are labeled "sample fixture" everywhere they appear.
  - They say nothing about full-corpus retrieval quality.

The fixture intentionally does NOT depend on document_processor, vector stores,
or any production src/ module. Similarity is computed with a lightweight
TF-IDF cosine function defined below, keeping the fixture self-contained.
"""

from typing import List, Dict

# ---------------------------------------------------------------------------
# 10 hand-built chunks (abbreviated 3GPP content)
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS: List[Dict] = [
    {
        "id": "chunk-001",
        "source": "38300-g10.docx",
        "text": (
            "The gNB (next generation NodeB) provides NR radio access to the UE. "
            "The gNB connects to the 5G Core (5GC) via the NG interface. "
            "In a disaggregated deployment the gNB consists of a gNB-CU (centralised unit) "
            "and one or more gNB-DU (distributed units). "
            "NG-RAN consists of a set of gNBs connected to the 5GC."
        ),
        "similarity": None,
    },
    {
        "id": "chunk-002",
        "source": "38401-h10.docx",
        "text": (
            "The F1 interface connects the gNB-CU to the gNB-DU. "
            "F1-C carries the F1 control plane (F1AP). "
            "F1-U carries the F1 user plane using GTP-U. "
            "The gNB-CU controls the gNB-DU through the F1 interface. "
            "Multiple gNB-DUs can connect to a single gNB-CU."
        ),
        "similarity": None,
    },
    {
        "id": "chunk-003",
        "source": "38473-h10.docx",
        "text": (
            "F1AP (F1 Application Protocol) is the application layer protocol for the F1 interface. "
            "F1AP procedures include F1 Setup, UE Context Management, and RRC message transfer. "
            "F1AP runs over SCTP/IP between the gNB-CU and gNB-DU."
        ),
        "similarity": None,
    },
    {
        "id": "chunk-004",
        "source": "38321-h10.docx",
        "text": (
            "The MAC (Medium Access Control) layer performs scheduling, HARQ retransmission, "
            "and logical channel multiplexing/demultiplexing. "
            "The MAC layer maps logical channels to transport channels. "
            "The scheduler in the gNB MAC controls uplink and downlink resource allocation. "
            "HARQ processes can run in parallel to improve throughput."
        ),
        "similarity": None,
    },
    {
        "id": "chunk-005",
        "source": "38331-h10.docx",
        "text": (
            "RRC (Radio Resource Control) handles connection establishment, maintenance, and release. "
            "RRC configures measurement reports and mobility procedures. "
            "RRC also carries the system information (SIB) broadcast by the gNB. "
            "RRC is defined in TS 38.331."
        ),
        "similarity": None,
    },
    {
        "id": "chunk-006",
        "source": "38300-g10.docx",
        "text": (
            "Xn-based handover allows a UE to move between gNBs without involving the AMF. "
            "The source gNB sends a Handover Request to the target gNB over the Xn interface. "
            "N2-based handover (via AMF) is used when the Xn interface is not available. "
            "Measurement reports from the UE trigger handover decisions in the gNB."
        ),
        "similarity": None,
    },
    {
        "id": "chunk-007",
        "source": "38211-h10.docx",
        "text": (
            "NR supports flexible numerology with subcarrier spacings of 15, 30, 60, 120, and 240 kHz. "
            "PDSCH (Physical Downlink Shared Channel) carries downlink shared channel data. "
            "PUSCH (Physical Uplink Shared Channel) carries uplink shared channel data. "
            "PBCH (Physical Broadcast Channel) carries the master information block (MIB). "
            "SSB (Synchronization Signal Block) consists of PSS, SSS, and PBCH."
        ),
        "similarity": None,
    },
    {
        "id": "chunk-008",
        "source": "38323-h10.docx",
        "text": (
            "PDCP (Packet Data Convergence Protocol) performs header compression using ROHC. "
            "PDCP also performs ciphering and integrity protection of user plane and control plane data. "
            "PDCP handles in-sequence delivery and duplicate detection on re-establishment."
        ),
        "similarity": None,
    },
    {
        "id": "chunk-009",
        "source": "38322-h10.docx",
        "text": (
            "The RLC (Radio Link Control) protocol operates in three modes: "
            "Transparent Mode (TM), Unacknowledged Mode (UM), and Acknowledged Mode (AM). "
            "RLC AM provides retransmission of lost PDUs. "
            "RLC UM provides one-directional data transfer without retransmission."
        ),
        "similarity": None,
    },
    {
        "id": "chunk-010",
        "source": "38410-h10.docx",
        "text": (
            "The NG interface connects the gNB to the 5G Core Network. "
            "NG-C (NG Control Plane) connects the gNB to the AMF using NGAP. "
            "NG-U (NG User Plane) connects the gNB to the UPF using GTP-U. "
            "NGAP (NG Application Protocol) handles UE context management and handover signalling."
        ),
        "similarity": None,
    },
]

# ---------------------------------------------------------------------------
# 6 fixture queries with ground truth
# ---------------------------------------------------------------------------
# Each query has:
#   relevant_sources   — spec tokens that must appear in retrieved chunks
#   expected_keywords  — keyword-coverage check (heuristic metric)
#   graded_relevance   — {spec_token: grade 0/1/2} for graded nDCG
#   expected_top_chunk — chunk id expected at rank 1 (used in smoke tests)

SAMPLE_QUERIES: List[Dict] = [
    {
        "id": "fix-001",
        "query": "What is gNB and how does it connect to the 5G core?",
        "relevant_sources": ["38300", "38401"],
        "expected_keywords": ["gnb", "ng", "5gc", "interface"],
        "graded_relevance": {"38300": 2, "38401": 1},
        "expected_top_chunk": "chunk-001",
        "notes": "Fixture: gNB definition — chunk-001 (ts_38.300) is the primary match.",
    },
    {
        "id": "fix-002",
        "query": "How does the F1 interface connect gNB-CU and gNB-DU?",
        "relevant_sources": ["38401", "38473"],
        "expected_keywords": ["f1", "cu", "du", "f1ap"],
        "graded_relevance": {"38401": 2, "38473": 2},
        "expected_top_chunk": "chunk-002",
        "notes": "Fixture: F1 interface — chunks 002 (38401) and 003 (38473) both relevant.",
    },
    {
        "id": "fix-003",
        "query": "What functions does the MAC layer perform in NR?",
        "relevant_sources": ["38321"],
        "expected_keywords": ["mac", "scheduling", "harq", "logical channel"],
        "graded_relevance": {"38321": 2},
        "expected_top_chunk": "chunk-004",
        "notes": "Fixture: MAC protocol — chunk-004 (ts_38.321) is the only relevant chunk.",
    },
    {
        "id": "fix-004",
        "query": "What does PDCP do and what compression does it use?",
        "relevant_sources": ["38323"],
        "expected_keywords": ["pdcp", "rohc", "header compression", "ciphering"],
        "graded_relevance": {"38323": 2},
        "expected_top_chunk": "chunk-008",
        "notes": "Fixture: PDCP — chunk-008 (ts_38.323) is the sole relevant chunk.",
    },
    {
        "id": "fix-005",
        "query": "What are the three modes of the RLC protocol?",
        "relevant_sources": ["38322"],
        "expected_keywords": ["rlc", "tm", "um", "am", "acknowledged"],
        "graded_relevance": {"38322": 2},
        "expected_top_chunk": "chunk-009",
        "notes": "Fixture: RLC modes — chunk-009 (ts_38.322) is the sole relevant chunk.",
    },
    {
        "id": "fix-006",
        "query": "What NR physical channels carry data and broadcast information?",
        "relevant_sources": ["38211"],
        "expected_keywords": ["pdsch", "pusch", "pbch", "physical channel"],
        "graded_relevance": {"38211": 2},
        "expected_top_chunk": "chunk-007",
        "notes": "Fixture: NR physical channels — chunk-007 (ts_38.211) is the primary match.",
    },
]
