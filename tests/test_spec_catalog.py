"""
Tests for src/core/spec_catalog.py helpers.

Currently covers infer_release_from_filename (Track C groundwork,
ADR-008): parsing the 3GPP release from archive-filename version
suffixes. Pure string logic — no index or services required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))



# ---------------------------------------------------------------------------
# infer_release_from_filename (Track C groundwork, ADR-008)
# ---------------------------------------------------------------------------

class TestInferReleaseFromFilename:
    """3GPP version-suffix release parsing: first version char is the
    release in base-36 (f=Rel-15 ... j=Rel-19)."""

    def test_rel17(self):
        from src.core.spec_catalog import infer_release_from_filename
        assert infer_release_from_filename("38300-h30.docx") == "Rel-17"

    def test_rel19(self):
        from src.core.spec_catalog import infer_release_from_filename
        assert infer_release_from_filename("38300-j10.docx") == "Rel-19"

    def test_rel16(self):
        from src.core.spec_catalog import infer_release_from_filename
        assert infer_release_from_filename("36331-g60.zip") == "Rel-16"

    def test_numeric_release(self):
        from src.core.spec_catalog import infer_release_from_filename
        assert infer_release_from_filename("36331-830.doc") == "Rel-8"

    def test_case_insensitive(self):
        from src.core.spec_catalog import infer_release_from_filename
        assert infer_release_from_filename("38300-H30.DOCX") == "Rel-17"

    def test_no_version_suffix(self):
        from src.core.spec_catalog import infer_release_from_filename
        assert infer_release_from_filename("38300.docx") is None

    def test_companion_suffix_not_a_version(self):
        from src.core.spec_catalog import infer_release_from_filename
        assert infer_release_from_filename("38300-cl.docx") is None

    def test_implausible_release_rejected(self):
        from src.core.spec_catalog import infer_release_from_filename
        # 'z' would be release 35 — outside the plausible window
        assert infer_release_from_filename("38300-z10.docx") is None
