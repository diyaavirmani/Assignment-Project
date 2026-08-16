"""
Query-time vocabulary expansion for 3GPP retrieval.

Why
---
The Phase 1 eval's three genuine retrieval misses (E1 interface, SA-vs-NSA,
NR-vs-LTE physical layer) share one root cause: the question's phrasing
diverges from spec terminology, and the embedding of the raw question lands
far from the chunks that define the answer. Appending the standard full forms
of any 3GPP abbreviations found in the query nudges the query embedding
toward spec vocabulary without discarding the user's own wording.

Provenance and integrity
------------------------
The vocabulary below is a general-purpose abbreviation table sourced from
3GPP TR 21.905 ("Vocabulary for 3GPP Specifications") and the abbreviation
clauses (§3) of the indexed specifications. It deliberately covers the
common RAN/Core/PHY vocabulary broadly rather than being tuned to the eval
golden set — expansion is applied uniformly to every query, and its effect
is measured by the eval harness, not assumed.

Usage
-----
    from src.core.query_expansion import expand_query
    expand_query("What is the difference between SA and NSA?")
    # -> "What is the difference between SA and NSA?
    #     (Standalone deployment; Non-Standalone deployment)"

Matching rules
--------------
* Short all-caps terms match case-sensitively on word boundaries, so "sa"
  in prose or "ran" as a verb never trigger.
* Mixed-case and hyphenated terms (gNB, NG-RAN, CU-CP) match
  case-insensitively on word boundaries.
* Hyphens count as boundaries, so "gNB" matches inside "gNB-CU". A longer
  entry and its embedded shorter entry may both fire; the resulting gloss
  is slightly redundant but harmless for embedding purposes.
"""
import re
from typing import Dict, List, Optional

# Abbreviation -> standard full form (TR 21.905 / spec §3 abbreviation clauses)
THREEGPP_VOCAB: Dict[str, str] = {
    # Deployment / architecture
    "SA":       "Standalone deployment",
    "NSA":      "Non-Standalone deployment",
    "EN-DC":    "E-UTRA NR Dual Connectivity",
    "MR-DC":    "Multi-Radio Dual Connectivity",
    "NG-RAN":   "Next Generation Radio Access Network",
    "E-UTRAN":  "Evolved Universal Terrestrial Radio Access Network",
    "RAN":      "Radio Access Network",
    "5GC":      "5G Core network",
    "EPC":      "Evolved Packet Core",
    "NR":       "New Radio",
    "LTE":      "Long Term Evolution",
    # Nodes
    "gNB":      "next generation NodeB base station",
    "eNB":      "evolved NodeB base station",
    "UE":       "User Equipment",
    "CU":       "Centralized Unit",
    "DU":       "Distributed Unit",
    "CU-CP":    "Centralized Unit Control Plane",
    "CU-UP":    "Centralized Unit User Plane",
    # Core network functions
    "AMF":      "Access and Mobility Management Function",
    "SMF":      "Session Management Function",
    "UPF":      "User Plane Function",
    "PCF":      "Policy Control Function",
    "UDM":      "Unified Data Management",
    "NSSF":     "Network Slice Selection Function",
    # Interfaces (definitions per the NG-RAN architecture specs)
    "F1":       "F1 interface between gNB-CU and gNB-DU",
    "E1":       "E1 interface between gNB-CU-CP and gNB-CU-UP",
    "Xn":       "Xn interface between NG-RAN nodes",
    "X2":       "X2 interface between eNBs",
    "S1":       "S1 interface between E-UTRAN and EPC",
    "NG interface": "NG interface between NG-RAN and 5G Core",
    # Protocols
    "RRC":      "Radio Resource Control protocol",
    "PDCP":     "Packet Data Convergence Protocol",
    "RLC":      "Radio Link Control protocol",
    "MAC":      "Medium Access Control protocol",
    "SDAP":     "Service Data Adaptation Protocol",
    "NAS":      "Non-Access Stratum",
    "NGAP":     "NG Application Protocol",
    "F1AP":     "F1 Application Protocol",
    "E1AP":     "E1 Application Protocol",
    "XnAP":     "Xn Application Protocol",
    # Physical layer / radio
    "PHY":      "physical layer",
    "HARQ":     "Hybrid Automatic Repeat Request retransmission",
    "RACH":     "Random Access Channel procedure",
    "PRACH":    "Physical Random Access Channel",
    "SSB":      "Synchronization Signal Block",
    "CSI":      "Channel State Information",
    "BWP":      "Bandwidth Part",
    "OFDM":     "Orthogonal Frequency Division Multiplexing",
    "MIMO":     "Multiple Input Multiple Output antenna",
    # QoS / slicing / bearers
    "QoS":      "Quality of Service",
    "QFI":      "QoS Flow Identifier",
    "NSSAI":    "Network Slice Selection Assistance Information",
    "PLMN":     "Public Land Mobile Network",
    "DRB":      "Data Radio Bearer",
    "SRB":      "Signalling Radio Bearer",
    "PDU":      "Protocol Data Unit",
    # Features
    "IAB":      "Integrated Access and Backhaul",
    "URLLC":    "Ultra-Reliable Low-Latency Communication",
    "eMBB":     "enhanced Mobile Broadband",
    "V2X":      "Vehicle-to-Everything",
}

# Word boundary: not adjacent to another letter/digit. Hyphens ARE
# boundaries, so "gNB" matches inside "gNB-CU" and "NG-RAN" matches after
# "5G ". Terms containing spaces ("NG interface") work unchanged.
_BOUNDARY_BEFORE = r"(?<![A-Za-z0-9])"
_BOUNDARY_AFTER = r"(?![A-Za-z0-9])"


def _term_pattern(term: str) -> "re.Pattern":
    return re.compile(_BOUNDARY_BEFORE + re.escape(term) + _BOUNDARY_AFTER)


def _term_matches(term: str, query: str) -> bool:
    """Case-sensitive for short all-caps terms; case-insensitive otherwise."""
    pattern = _BOUNDARY_BEFORE + re.escape(term) + _BOUNDARY_AFTER
    if term.isupper() and len(term) <= 5:
        return re.search(pattern, query) is not None
    return re.search(pattern, query, re.IGNORECASE) is not None


def expand_query(query: str, vocab: Optional[Dict[str, str]] = None) -> str:
    """Append the full forms of any known 3GPP abbreviations in the query.

    The original query text is kept verbatim; matched expansions are added
    as a single trailing parenthetical gloss. Queries containing no known
    abbreviations are returned unchanged.

    Args:
        query: The user's natural-language question.
        vocab: Override the abbreviation table (defaults to THREEGPP_VOCAB).

    Returns:
        The expanded query string, or the original query if nothing matched.
    """
    table = THREEGPP_VOCAB if vocab is None else vocab
    expansions: List[str] = []
    for term, full_form in table.items():
        if _term_matches(term, query) and full_form not in expansions:
            expansions.append(full_form)
    if not expansions:
        return query
    return f"{query} ({'; '.join(expansions)})"
