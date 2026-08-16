"""Tests for src/core/answer_verifier.py."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.answer_verifier import (
    AnswerVerifier,
    REASON_CONTRADICTION,
    REASON_EMPTY_ANSWER,
    REASON_INVALID_CITATION,
    REASON_MISSING_CITATIONS,
    REASON_UNSUPPORTED,
    REASON_VERIFICATION_ERROR,
    REASON_VERIFIED,
)


DOCS = [
    {
        "text": "The gNB-CU is a logical node hosting RRC and PDCP.",
        "source": "38300-g30.docx",
        "chunk_index": 0,
        "similarity": 0.92,
        "spec_number": "38.300",
        "spec_title": "NR overall description",
    },
    {
        "text": "The gNB-DU connects to the gNB-CU through the F1 interface.",
        "source": "38401-g30.docx",
        "chunk_index": 1,
        "similarity": 0.88,
        "spec_number": "38.401",
        "spec_title": "NG-RAN architecture",
    },
]


def _response(payload):
    content = json.dumps(payload) if not isinstance(payload, str) else payload
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _client(payload):
    client = MagicMock()
    client.chat.completions.create.return_value = _response(payload)
    return client


class TestAnswerVerifier:
    def test_supported_f1_interface_answer_passes(self):
        verifier = AnswerVerifier(
            model="gpt-test",
            client=_client(
                {
                    "claims": [
                        {
                            "claim": "The F1 interface connects the gNB-CU and gNB-DU.",
                            "status": "SUPPORTED",
                            "source_ids": ["S2"],
                        }
                    ]
                }
            ),
        )

        result = verifier.verify(
            "What is the F1 interface used for?",
            DOCS,
            "The F1 interface connects the gNB-CU and gNB-DU [S2].",
        )

        assert result.passed is True
        assert result.reason == REASON_VERIFIED
        assert result.citation_valid is True
        assert result.cited_sources == ["S2"]

    def test_fully_supported_answer_passes(self):
        verifier = AnswerVerifier(
            model="gpt-test",
            client=_client(
                {
                    "claims": [
                        {
                            "claim": "The gNB-CU hosts RRC and PDCP.",
                            "status": "SUPPORTED",
                            "source_ids": ["S1"],
                        }
                    ]
                }
            ),
        )

        result = verifier.verify(
            "What does the gNB-CU host?",
            DOCS,
            "The gNB-CU hosts RRC and PDCP [S1].",
        )

        assert result.passed is True
        assert result.reason == REASON_VERIFIED
        assert result.total_claims == 1
        assert result.supported_claims == 1
        assert result.citation_valid is True
        assert result.cited_sources == ["S1"]

    def test_one_unsupported_claim_fails(self):
        verifier = AnswerVerifier(
            model="gpt-test",
            client=_client(
                {
                    "claims": [
                        {
                            "claim": "The gNB-CU hosts RRC.",
                            "status": "SUPPORTED",
                            "source_ids": ["S1"],
                        },
                        {
                            "claim": "The gNB-CU performs billing.",
                            "status": "UNSUPPORTED",
                            "source_ids": ["S1"],
                        },
                    ]
                }
            ),
        )

        result = verifier.verify(
            "What does the gNB-CU do?",
            DOCS,
            "The gNB-CU hosts RRC and performs billing [S1].",
        )

        assert result.passed is False
        assert result.reason == REASON_UNSUPPORTED
        assert result.unsupported_claims == ["The gNB-CU performs billing."]

    def test_contradicted_claim_fails(self):
        verifier = AnswerVerifier(
            model="gpt-test",
            client=_client(
                {
                    "claims": [
                        {
                            "claim": "The gNB-DU is not connected through F1.",
                            "status": "CONTRADICTED",
                            "source_ids": ["S2"],
                        }
                    ]
                }
            ),
        )

        result = verifier.verify(
            "How does gNB-DU connect?",
            DOCS,
            "The gNB-DU is not connected through F1 [S2].",
        )

        assert result.passed is False
        assert result.reason == REASON_CONTRADICTION
        assert result.contradicted_claims == ["The gNB-DU is not connected through F1."]

    def test_misleading_e1_du_upf_premise_remains_blocked(self):
        docs = [
            {
                "text": "The E1 interface connects the gNB-CU-CP and gNB-CU-UP.",
                "source": "38401-g30.docx",
                "chunk_index": 0,
                "similarity": 0.91,
                "spec_number": "38.401",
                "spec_title": "NG-RAN architecture",
            }
        ]
        verifier = AnswerVerifier(
            model="gpt-test",
            client=_client(
                {
                    "claims": [
                        {
                            "claim": "The E1 interface connects gNB-DU to the UPF.",
                            "status": "CONTRADICTED",
                            "source_ids": ["S1"],
                        }
                    ]
                }
            ),
        )

        result = verifier.verify(
            "Explain why the E1 interface connects gNB-DU to the UPF.",
            docs,
            "The E1 interface connects gNB-DU to the UPF [S1].",
        )

        assert result.passed is False
        assert result.reason == REASON_CONTRADICTION
        assert result.contradicted_claims == [
            "The E1 interface connects gNB-DU to the UPF."
        ]

    def test_valid_citations_pass_deterministic_validation(self):
        verifier = AnswerVerifier(enabled=False)

        result = verifier.validate_citations("A claim [S1] and another [S2].", DOCS)

        assert result.passed is True
        assert result.citation_valid is True
        assert result.cited_sources == ["S1", "S2"]

    def test_invented_citation_fails(self):
        verifier = AnswerVerifier(enabled=False)

        result = verifier.validate_citations("A claim [S99].", DOCS)

        assert result.passed is False
        assert result.reason == REASON_INVALID_CITATION
        assert result.invalid_citations == ["S99"]

    def test_no_citation_in_factual_answer_fails(self):
        verifier = AnswerVerifier(enabled=False)

        result = verifier.validate_citations("The gNB-CU hosts RRC.", DOCS)

        assert result.passed is False
        assert result.reason == REASON_MISSING_CITATIONS

    def test_empty_generated_answer_fails(self):
        verifier = AnswerVerifier(enabled=False)

        result = verifier.validate_citations("   ", DOCS)

        assert result.passed is False
        assert result.reason == REASON_EMPTY_ANSWER

    def test_verifier_api_exception_fails_closed(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("boom")
        verifier = AnswerVerifier(model="gpt-test", client=client)

        result = verifier.verify(
            "What does the gNB-CU host?",
            DOCS,
            "The gNB-CU hosts RRC and PDCP [S1].",
        )

        assert result.passed is False
        assert result.reason == REASON_VERIFICATION_ERROR

    def test_invalid_structured_verifier_response_fails_closed(self):
        verifier = AnswerVerifier(model="gpt-test", client=_client({"not_claims": []}))

        result = verifier.verify(
            "What does the gNB-CU host?",
            DOCS,
            "The gNB-CU hosts RRC and PDCP [S1].",
        )

        assert result.passed is False
        assert result.reason == REASON_VERIFICATION_ERROR

    def test_multiple_valid_citations_pass(self):
        verifier = AnswerVerifier(
            model="gpt-test",
            client=_client(
                {
                    "claims": [
                        {
                            "claim": "The gNB-CU hosts RRC and PDCP.",
                            "status": "SUPPORTED",
                            "source_ids": ["S1"],
                        },
                        {
                            "claim": "The gNB-DU connects via F1.",
                            "status": "SUPPORTED",
                            "source_ids": ["S2"],
                        },
                    ]
                }
            ),
        )

        result = verifier.verify(
            "Explain CU and DU.",
            DOCS,
            "The gNB-CU hosts RRC and PDCP [S1]. The gNB-DU connects via F1 [S2].",
        )

        assert result.passed is True
        assert result.reason == REASON_VERIFIED
        assert result.cited_sources == ["S1", "S2"]

    def test_missing_api_key_raises_clear_error(self):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            AnswerVerifier(model="gpt-test", api_key="")
