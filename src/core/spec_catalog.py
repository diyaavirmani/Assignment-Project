"""
3GPP Specification Catalog

Defines the curated set of 3GPP specs supported by the RAG system.
Each entry carries metadata used for chunk tagging and UI filtering.

Domains:  RAN  — Radio Access Network
          CORE — Core Network / EPC / 5GC

Generations: 5G  — 5G NR / NG-RAN / 5GC
             LTE — LTE / E-UTRAN / EPC

Corpus-config seam
------------------
This file is the source of truth for the 3GPP catalog data.
``src/core/corpus_config.py`` wraps it in a ``CorpusConfig`` and exposes the
corpus-agnostic interface used by the rest of the pipeline.  To access the
full config (text cleaner, filter dimensions, factory) import from there:

    from src.core.corpus_config import build_3gpp_corpus_config, DEFAULT_CORPUS

Nothing in this file changes when a new corpus is added; a new corpus simply
creates its own CorpusConfig without touching the 3GPP data here.
"""
import re
from typing import List, Optional, TypedDict


class SpecEntry(TypedDict):
    spec_number: str    # e.g. "38.300"
    series: str         # e.g. "38_series"  (matches 3GPP FTP path)
    title: str
    domain: str         # "RAN" | "CORE"
    generation: str     # "5G" | "LTE"
    description: str


CATALOG: List[SpecEntry] = [
    # -------------------------------------------------------------------------
    # 5G NR RAN — 38.xxx series
    # -------------------------------------------------------------------------
    {
        "spec_number": "38.211",
        "series": "38_series",
        "title": "NR; Physical channels and modulation",
        "domain": "RAN",
        "generation": "5G",
        "description": "Physical channel structure, modulation, and signal generation for 5G NR",
    },
    {
        "spec_number": "38.212",
        "series": "38_series",
        "title": "NR; Multiplexing and channel coding",
        "domain": "RAN",
        "generation": "5G",
        "description": "Transport channel coding, HARQ, and multiplexing for 5G NR",
    },
    {
        "spec_number": "38.213",
        "series": "38_series",
        "title": "NR; Physical layer procedures for control",
        "domain": "RAN",
        "generation": "5G",
        "description": "Physical layer control procedures: PDCCH, PRACH, power control for 5G NR",
    },
    {
        "spec_number": "38.214",
        "series": "38_series",
        "title": "NR; Physical layer procedures for data",
        "domain": "RAN",
        "generation": "5G",
        "description": "Physical layer data procedures: PDSCH, PUSCH, link adaptation for 5G NR",
    },
    {
        "spec_number": "38.215",
        "series": "38_series",
        "title": "NR; Physical layer measurements",
        "domain": "RAN",
        "generation": "5G",
        "description": "Physical layer measurement definitions for 5G NR",
    },
    {
        "spec_number": "38.300",
        "series": "38_series",
        "title": "NR; NR and NG-RAN Overall Description; Stage 2",
        "domain": "RAN",
        "generation": "5G",
        "description": "Overall NR and NG-RAN architecture, protocols, and functionality description",
    },
    {
        "spec_number": "38.321",
        "series": "38_series",
        "title": "NR; Medium Access Control (MAC) protocol specification",
        "domain": "RAN",
        "generation": "5G",
        "description": "NR MAC layer: scheduling, HARQ, random access procedures",
    },
    {
        "spec_number": "38.322",
        "series": "38_series",
        "title": "NR; Radio Link Control (RLC) protocol specification",
        "domain": "RAN",
        "generation": "5G",
        "description": "NR RLC protocol: TM, UM, and AM modes",
    },
    {
        "spec_number": "38.323",
        "series": "38_series",
        "title": "NR; Packet Data Convergence Protocol (PDCP) specification",
        "domain": "RAN",
        "generation": "5G",
        "description": "NR PDCP: header compression, ciphering, integrity protection",
    },
    {
        "spec_number": "38.331",
        "series": "38_series",
        "title": "NR; Radio Resource Control (RRC) protocol specification",
        "domain": "RAN",
        "generation": "5G",
        "description": "NR RRC: connection management, mobility, measurement reporting",
    },
    {
        "spec_number": "38.401",
        "series": "38_series",
        "title": "NG-RAN; Architecture description",
        "domain": "RAN",
        "generation": "5G",
        "description": "NG-RAN architecture: gNB-CU/DU split, F1/E1/Xn/NG interfaces",
    },
    {
        "spec_number": "38.410",
        "series": "38_series",
        "title": "NG-RAN; NG general aspects and principles",
        "domain": "RAN",
        "generation": "5G",
        "description": "NG interface general aspects between NG-RAN and 5GC",
    },
    {
        "spec_number": "38.413",
        "series": "38_series",
        "title": "NG-RAN; NG Application Protocol (NGAP)",
        "domain": "RAN",
        "generation": "5G",
        "description": "NGAP protocol between gNB and AMF: handover, paging, UE context management",
    },
    {
        "spec_number": "38.420",
        "series": "38_series",
        "title": "NG-RAN; Xn general aspects and principles",
        "domain": "RAN",
        "generation": "5G",
        "description": "Xn interface general aspects between neighbouring gNBs",
    },
    {
        "spec_number": "38.423",
        "series": "38_series",
        "title": "NG-RAN; Xn Application Protocol (XnAP)",
        "domain": "RAN",
        "generation": "5G",
        "description": "XnAP protocol for gNB-to-gNB communication and handover signalling",
    },
    {
        "spec_number": "38.470",
        "series": "38_series",
        "title": "NG-RAN; F1 general aspects and principles",
        "domain": "RAN",
        "generation": "5G",
        "description": "F1 interface general aspects between gNB-CU and gNB-DU",
    },
    {
        "spec_number": "38.473",
        "series": "38_series",
        "title": "NG-RAN; F1 Application Protocol (F1AP)",
        "domain": "RAN",
        "generation": "5G",
        "description": "F1AP protocol for CU-DU split communication in NG-RAN",
    },
    {
        "spec_number": "38.460",
        "series": "38_series",
        "title": "NG-RAN; E1 general aspects and principles",
        "domain": "RAN",
        "generation": "5G",
        "description": "E1 interface between gNB-CU-CP and gNB-CU-UP",
    },
    {
        "spec_number": "38.463",
        "series": "38_series",
        "title": "NG-RAN; E1 Application Protocol (E1AP)",
        "domain": "RAN",
        "generation": "5G",
        "description": "E1AP protocol for CP-UP split within the gNB-CU",
    },
    # -------------------------------------------------------------------------
    # LTE RAN — 36.xxx series
    # -------------------------------------------------------------------------
    {
        "spec_number": "36.211",
        "series": "36_series",
        "title": "E-UTRA; Physical channels and modulation",
        "domain": "RAN",
        "generation": "LTE",
        "description": "Physical channel structure and modulation for LTE/E-UTRA",
    },
    {
        "spec_number": "36.212",
        "series": "36_series",
        "title": "E-UTRA; Multiplexing and channel coding",
        "domain": "RAN",
        "generation": "LTE",
        "description": "Transport channel coding and multiplexing for LTE",
    },
    {
        "spec_number": "36.213",
        "series": "36_series",
        "title": "E-UTRA; Physical layer procedures",
        "domain": "RAN",
        "generation": "LTE",
        "description": "Physical layer procedures for LTE: PDCCH, power control, measurements",
    },
    {
        "spec_number": "36.300",
        "series": "36_series",
        "title": "E-UTRAN; Overall description; Stage 2",
        "domain": "RAN",
        "generation": "LTE",
        "description": "Overall E-UTRAN architecture: eNB, X2 and S1 interfaces",
    },
    {
        "spec_number": "36.321",
        "series": "36_series",
        "title": "E-UTRA; Medium Access Control (MAC) protocol specification",
        "domain": "RAN",
        "generation": "LTE",
        "description": "LTE MAC layer: scheduling, HARQ, random access procedures",
    },
    {
        "spec_number": "36.322",
        "series": "36_series",
        "title": "E-UTRA; Radio Link Control (RLC) protocol specification",
        "domain": "RAN",
        "generation": "LTE",
        "description": "LTE RLC protocol specification: TM, UM, AM modes",
    },
    {
        "spec_number": "36.323",
        "series": "36_series",
        "title": "E-UTRA; Packet Data Convergence Protocol (PDCP) specification",
        "domain": "RAN",
        "generation": "LTE",
        "description": "LTE PDCP: header compression and ciphering",
    },
    {
        "spec_number": "36.331",
        "series": "36_series",
        "title": "E-UTRA; Radio Resource Control (RRC) protocol specification",
        "domain": "RAN",
        "generation": "LTE",
        "description": "LTE RRC: connection management, mobility, measurement procedures",
    },
    {
        "spec_number": "36.401",
        "series": "36_series",
        "title": "E-UTRAN; Architecture description",
        "domain": "RAN",
        "generation": "LTE",
        "description": "E-UTRAN architecture description",
    },
    {
        "spec_number": "36.413",
        "series": "36_series",
        "title": "E-UTRAN; S1 Application Protocol (S1AP)",
        "domain": "RAN",
        "generation": "LTE",
        "description": "S1AP protocol between eNB and MME",
    },
    {
        "spec_number": "36.423",
        "series": "36_series",
        "title": "E-UTRAN; X2 Application Protocol (X2AP)",
        "domain": "RAN",
        "generation": "LTE",
        "description": "X2AP protocol for eNB-to-eNB communication and handover",
    },
    # -------------------------------------------------------------------------
    # 5G Core — 23.xxx / 29.xxx series
    # -------------------------------------------------------------------------
    {
        "spec_number": "23.501",
        "series": "23_series",
        "title": "System Architecture for the 5G System (5GS); Stage 2",
        "domain": "CORE",
        "generation": "5G",
        "description": "5GS architecture: AMF, SMF, UPF, PCF, AUSF, NRF and their interfaces",
    },
    {
        "spec_number": "23.502",
        "series": "23_series",
        "title": "Procedures for the 5G System (5GS); Stage 2",
        "domain": "CORE",
        "generation": "5G",
        "description": "5GS procedures: registration, session management, mobility, policy control",
    },
    {
        "spec_number": "23.503",
        "series": "23_series",
        "title": "Policy and Charging Control Framework for the 5G System (5GS); Stage 2",
        "domain": "CORE",
        "generation": "5G",
        "description": "5G PCC framework: PCF, policy rules, QoS flows",
    },
    {
        "spec_number": "29.500",
        "series": "29_series",
        "title": "5G System; Technical Realization of Service Based Architecture; Stage 3",
        "domain": "CORE",
        "generation": "5G",
        "description": "5GC SBA: HTTP/2, JSON, OpenAPI for NF service interfaces",
    },
    # -------------------------------------------------------------------------
    # LTE Core / EPC — 23.xxx / 29.xxx series
    # -------------------------------------------------------------------------
    {
        "spec_number": "23.401",
        "series": "23_series",
        "title": "GPRS enhancements for E-UTRAN access",
        "domain": "CORE",
        "generation": "LTE",
        "description": "EPC architecture: MME, SGW, PGW, HSS and S1/S11/S5 interfaces",
    },
    {
        "spec_number": "23.402",
        "series": "23_series",
        "title": "Architecture enhancements for non-3GPP accesses",
        "domain": "CORE",
        "generation": "LTE",
        "description": "Non-3GPP access to EPC including WiFi/WLAN integration",
    },
    {
        "spec_number": "29.274",
        "series": "29_series",
        "title": "3GPP EPS; Evolved GTP for Control plane (GTPv2-C)",
        "domain": "CORE",
        "generation": "LTE",
        "description": "GTPv2-C protocol on S11, S5/S8 interfaces in EPC",
    },
]


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_spec(spec_number: str) -> Optional[SpecEntry]:
    """Return the catalog entry for a spec number, or None if not found."""
    for entry in CATALOG:
        if entry["spec_number"] == spec_number:
            return entry
    return None


def get_specs_by_filter(
    domain: Optional[str] = None,
    generation: Optional[str] = None,
) -> List[SpecEntry]:
    """Return all specs matching the given domain and/or generation."""
    results = CATALOG
    if domain:
        results = [s for s in results if s["domain"] == domain]
    if generation:
        results = [s for s in results if s["generation"] == generation]
    return results


def infer_spec_from_filename(filename: str) -> Optional[SpecEntry]:
    """Infer catalog entry from a filename.

    Looks for patterns like '38300', '38.300', '36331' anywhere in the
    filename (case-insensitive). Returns the first matching entry.
    """
    fname = filename.lower()
    for entry in CATALOG:
        spec_num = entry["spec_number"]          # "38.300"
        spec_nodot = spec_num.replace(".", "")   # "38300"
        if spec_nodot in fname or spec_num in fname:
            return entry
    return None


def get_ftp_url(entry: SpecEntry) -> str:
    """Return the 3GPP FTP archive directory URL for a spec."""
    return (
        f"https://www.3gpp.org/ftp/Specs/archive/"
        f"{entry['series']}/{entry['spec_number']}/"
    )


# 3GPP archive filenames encode the spec version as -<XYZ> where X, Y, Z
# are base-36 digits (major.technical.editorial) and the major digit is the
# release: '8' = Rel-8, 'a' = Rel-10, 'f' = Rel-15, 'g' = Rel-16,
# 'h' = Rel-17, 'i' = Rel-18, 'j' = Rel-19. Example: 38300-h30.docx is
# TS 38.300 version 17.3.0 (Rel-17).
_VERSION_SUFFIX = re.compile(r"-([0-9a-z])([0-9a-z]{2})\.[a-z]+$")

# Plausible release window; anything outside is treated as not-a-version
# (guards against companion-file suffixes accidentally matching).
_MIN_RELEASE, _MAX_RELEASE = 5, 24


def infer_release_from_filename(filename: str) -> Optional[str]:
    """Infer the 3GPP release from a spec archive filename.

    Returns e.g. ``"Rel-17"`` for ``38300-h30.docx``, or None when the
    filename carries no recognizable version suffix. Stored as chunk
    metadata by the index builder so every citation can name its release —
    the first building block of release-delta answers (see ADR-008).
    """
    m = _VERSION_SUFFIX.search(filename.lower())
    if not m:
        return None
    release = int(m.group(1), 36)
    if not (_MIN_RELEASE <= release <= _MAX_RELEASE):
        return None
    return f"Rel-{release}"


def catalog_summary() -> str:
    """Return a human-readable summary of the catalog."""
    by_gen_domain: dict = {}
    for entry in CATALOG:
        key = f"{entry['generation']} {entry['domain']}"
        by_gen_domain.setdefault(key, []).append(entry["spec_number"])

    lines = [f"3GPP Spec Catalog — {len(CATALOG)} specs total"]
    for key in sorted(by_gen_domain):
        specs = ", ".join(sorted(by_gen_domain[key]))
        lines.append(f"  {key}: {specs}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(catalog_summary())
