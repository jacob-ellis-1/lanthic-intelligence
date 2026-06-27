from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


THIS_FILE = Path(__file__).resolve()
EXTRACTION_DIR = THIS_FILE.parent
SRC_DIR = EXTRACTION_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

for path in (EXTRACTION_DIR, SRC_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import extract

class FakeBlock:
    def __init__(self, block_id="blk_1", text="Lynas operates the Gebeng processing facility."):
        self.block_id = block_id
        self.document_id = "doc_1"
        self.block_type = "text"
        self.source_url = "https://example.com/source"
        self.metadata = {}
        self.text = text

    def to_text(self):
        return self.text

    def to_dict(self):
        return {
            "block_id": self.block_id,
            "document_id": self.document_id,
            "block_type": self.block_type,
            "source_url": self.source_url,
            "metadata": self.metadata,
            "text": self.text,
        }


class FakeDocument:
    def __init__(self):
        self.document_id = "doc_1"
        self.source_type = "html_webpage"
        self.metadata = type("Metadata", (), {
            "title": "Fake source",
            "author": None,
            "publisher": "Example",
            "published_at": "2026-01-01",
            "source_url": "https://example.com/source",
            "canonical_url": "https://example.com/source",
        })()
        self.credibility = type("Credibility", (), {
            "score": 0.9,
            "tier": "high",
            "rationale": "test source",
        })()
        self.blocks = [FakeBlock()]

    def ensure_blocks(self):
        return self.blocks


def fake_llm_json(*args, **kwargs):
    return {
        "entities": [
            {
                "entity_id": "e1",
                "canonical_name": "Lynas",
                "entity_type": "company",
                "aliases": [],
                "description": "Company named in the evidence.",
                "attributes": {},
                "temporal": {
                    "event_date": None,
                    "valid_from": None,
                    "valid_to": None,
                },
                "evidence": [
                    {
                        "evidence_id": "blk_1",
                        "quote": "Lynas operates the Gebeng processing facility.",
                    }
                ],
            },
            {
                "entity_id": "e2",
                "canonical_name": "Gebeng processing facility",
                "entity_type": "facility",
                "aliases": [],
                "description": "Facility named in the evidence.",
                "attributes": {},
                "temporal": {
                    "event_date": None,
                    "valid_from": None,
                    "valid_to": None,
                },
                "evidence": [
                    {
                        "evidence_id": "blk_1",
                        "quote": "Lynas operates the Gebeng processing facility.",
                    }
                ],
            },
        ],
        "relations": [
            {
                "relation_id": "r1",
                "subject_id": "e1",
                "relation_type": "operates",
                "object_id": "e2",
                "description": "Lynas operates the facility.",
                "temporal": {
                    "event_date": None,
                    "valid_from": None,
                    "valid_to": None,
                },
                "extraction_confidence": 0.9,
                "attributes": {},
                "evidence": [
                    {
                        "evidence_id": "blk_1",
                        "quote": "Lynas operates the Gebeng processing facility.",
                    }
                ],
            }
        ],
    }


class ExtractInputTests(unittest.TestCase):
    def test_loaded_document_json_exposes_expected_document_interface(self):
        raw = {
            "document_id": "doc_1",
            "source_type": "text_file",
            "metadata": {
                "title": "Test document",
                "author": None,
                "publisher": "Test publisher",
                "published_at": "2026-01-01",
                "source_url": "file:///tmp/test.txt",
                "canonical_url": "file:///tmp/test.txt",
            },
            "credibility": {
                "score": 0.5,
                "tier": "unknown",
                "rationale": "",
            },
            "blocks": [
                {
                    "block_id": "blk_1",
                    "document_id": "doc_1",
                    "block_type": "text",
                    "source_url": "file:///tmp/test.txt",
                    "metadata": {},
                    "text": "Lynas operates the Gebeng processing facility.",
                }
            ],
        }

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "document.json"
            path.write_text(json.dumps(raw), encoding="utf-8")

            document = extract.document_from_json(path)

        self.assertEqual(document.document_id, "doc_1")
        self.assertEqual(document.metadata.title, "Test document")
        self.assertEqual(len(document.ensure_blocks()), 1)
        self.assertEqual(document.ensure_blocks()[0].block_id, "blk_1")
        self.assertEqual(
            document.ensure_blocks()[0].to_text(),
            "Lynas operates the Gebeng processing facility.",
        )

    def test_extract_from_input_produces_same_output_shape_as_extraction_pipeline(self):
        raw = {
            "document_id": "doc_1",
            "source_type": "text_file",
            "metadata": {
                "title": "Test document",
                "author": None,
                "publisher": "Test publisher",
                "published_at": "2026-01-01",
                "source_url": "file:///tmp/test.txt",
                "canonical_url": "file:///tmp/test.txt",
            },
            "credibility": {
                "score": 0.5,
                "tier": "unknown",
                "rationale": "",
            },
            "blocks": [
                {
                    "block_id": "blk_1",
                    "document_id": "doc_1",
                    "block_type": "text",
                    "source_url": "file:///tmp/test.txt",
                    "metadata": {},
                    "text": "Lynas operates the Gebeng processing facility.",
                }
            ],
        }

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "document.json"
            path.write_text(json.dumps(raw), encoding="utf-8")

            with patch.object(extract, "call_llm_json", side_effect=fake_llm_json):
                result = extract.extract_from_input(
                    path,
                    model="test-model",
                    include_evidence_text=True,
                )

        self.assertIn("document", result)
        self.assertIn("source_url", result)
        self.assertIn("source_id", result)
        self.assertIn("evidence_manifest", result)
        self.assertIn("entities", result)
        self.assertIn("relations", result)
        self.assertIn("extraction", result)
        self.assertIn("evidence_store", result)

        self.assertEqual(result["source_id"], "doc_1")
        self.assertEqual(result["source_url"], "file:///tmp/test.txt")
        self.assertEqual(len(result["entities"]), 2)
        self.assertEqual(len(result["relations"]), 1)
        self.assertEqual(result["entities"][0]["canonical_name"], "Lynas")
        self.assertEqual(result["relations"][0]["relation_type"], "operates")

    def test_url_mode_still_uses_existing_url_loader_path(self):
        with patch.object(extract, "html_webpage_document_from_url", return_value=FakeDocument()) as loader:
            with patch.object(extract, "call_llm_json", side_effect=fake_llm_json):
                result = extract.extract_from_url(
                    "https://example.com/source",
                    model="test-model",
                    include_evidence_text=True,
                )

        loader.assert_called_once_with(
            "https://example.com/source",
            use_ollama_chunking=False,
        )

        self.assertEqual(result["source_id"], "doc_1")
        self.assertEqual(result["source_url"], "https://example.com/source")
        self.assertEqual(len(result["entities"]), 2)
        self.assertEqual(len(result["relations"]), 1)

    def test_main_uses_input_mode_when_input_argument_is_present(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            input_path = tmp / "document.json"
            output_path = tmp / "extract.json"

            input_path.write_text(json.dumps({
                "document_id": "doc_1",
                "source_type": "text_file",
                "metadata": {
                    "title": "Test document",
                    "author": None,
                    "publisher": "Test publisher",
                    "published_at": "2026-01-01",
                    "source_url": "file:///tmp/test.txt",
                    "canonical_url": "file:///tmp/test.txt",
                },
                "credibility": {
                    "score": 0.5,
                    "tier": "unknown",
                    "rationale": "",
                },
                "blocks": [
                    {
                        "block_id": "blk_1",
                        "document_id": "doc_1",
                        "block_type": "text",
                        "source_url": "file:///tmp/test.txt",
                        "metadata": {},
                        "text": "Lynas operates the Gebeng processing facility.",
                    }
                ],
            }), encoding="utf-8")

            argv = [
                "extract.py",
                "--input", str(input_path),
                "--output", str(output_path),
                "--model", "test-model",
                "--include-evidence-text",
            ]

            with patch.object(sys, "argv", argv):
                with patch.object(extract, "call_llm_json", side_effect=fake_llm_json):
                    exit_code = extract.main()

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())

            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["source_id"], "doc_1")
            self.assertEqual(len(result["entities"]), 2)
            self.assertEqual(len(result["relations"]), 1)


if __name__ == "__main__":
    unittest.main()