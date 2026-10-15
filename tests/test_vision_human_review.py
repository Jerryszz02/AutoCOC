from __future__ import annotations

from http.server import HTTPServer
import json
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import threading
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pytest

from training.vision.data import DatasetError, _pixel_hash, read_jsonl, sha256, validate, write_jsonl
from training.vision.human_review import SceneReview, make_handler


def fixture_dataset(base: Path) -> tuple[Path, Path]:
    root = base / "source"
    (root / "images").mkdir(parents=True, exist_ok=True)
    image = root / "images" / "one.png"
    pixels = np.full((12, 16, 3), 64, dtype=np.uint8)
    assert cv2.imwrite(str(image), pixels)
    row = {"schema_version": 1, "sample_id": "S1", "image_path": "images/one.png",
           "file_sha256": sha256(image), "pixel_sha256": _pixel_hash(pixels),
           "width": 16, "height": 12, "source_domain": "local", "source_ref": "fixture",
           "source_session": "session-one", "battle_id": None, "layout_group_id": None,
           "leakage_group_id": "group-one", "parent_sample_id": None,
           "client_version": "18.647.2", "review_status": "reviewed",
           "reviewer": "agent", "review_method": "agent_visual", "review_evidence": "prototype image inspection",
           "split": "unassigned"}
    annotation = {"schema_version": 1, "sample_id": "S1", "scene": {
        "label": "village", "complete": True, "review_status": "reviewed",
        "reviewer": "agent", "review_method": "agent_visual", "review_evidence": "prototype image inspection"}}
    write_jsonl(root / "manifest.jsonl", [row])
    write_jsonl(root / "annotations.jsonl", [annotation])
    return root, base / "human-verdicts.jsonl"


def test_append_only_history_and_export_only_latest_confirmed_scene():
    with TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, journal = fixture_dataset(base)
        review = SceneReview(root, journal)
        first = review.append("S1", "confirm", "village", "实际用户")
        assert first["sequence"] == 1
        second = review.append("S1", "correct", "training", "实际用户")
        assert second["previous_sha256"] == first["event_sha256"]
        assert read_jsonl(journal)[0]["action"] == "confirm"
        review = SceneReview(root, journal)
        result = review.export(base / "export")
        assert result["confirmed_samples"] == 1
        exported_manifest = read_jsonl(base / "export" / "manifest.jsonl")
        exported_annotations = read_jsonl(base / "export" / "annotations.jsonl")
        assert exported_manifest[0]["review_method"] == "human"
        assert exported_manifest[0]["source_review"]["review_method"] == "agent_visual"
        assert exported_annotations[0]["scene"]["label"] == "training"
        assert exported_annotations[0]["scene"]["source_agent_scene"]["label"] == "village"
        assert len(read_jsonl(base / "export" / "human-verdicts.jsonl")) == 2
        assert validate(base / "export", exported_manifest, exported_annotations)["valid"]
        with pytest.raises(DatasetError, match="overwrite"):
            review.export(base / "export")


def test_skip_revokes_confirmation_and_never_marks_human():
    with TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, journal = fixture_dataset(base)
        review = SceneReview(root, journal)
        review.append("S1", "confirm", "village", "实际用户")
        review.append("S1", "skip", None, "实际用户")
        assert review.latest()["S1"]["action"] == "skip"
        with pytest.raises(DatasetError, match="No explicit"):
            review.export(base / "unused-export")
        assert read_jsonl(root / "manifest.jsonl")[0]["review_method"] == "agent_visual"
        assert len(read_jsonl(journal)) == 2


def test_source_and_journal_tampering_rejected():
    with TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, journal = fixture_dataset(base)
        review = SceneReview(root, journal)
        review.append("S1", "confirm", "village", "实际用户")
        (root / "images" / "one.png").write_bytes(b"changed")
        with pytest.raises(DatasetError, match="SHA-256 mismatch"):
            review.export(base / "unused-export")
        root, journal = fixture_dataset(base)
        event = read_jsonl(journal)[0]
        event["reviewer"] = "forged"
        write_jsonl(journal, [event])
        with pytest.raises(DatasetError, match="event hash mismatch"):
            SceneReview(root, journal)


@pytest.mark.parametrize("changed_file", ["manifest.jsonl", "annotations.jsonl"])
def test_export_rejects_changed_source_record(changed_file: str):
    with TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, journal = fixture_dataset(base)
        review = SceneReview(root, journal)
        review.append("S1", "confirm", "village", "实际用户")
        target = root / changed_file
        rows = read_jsonl(target)
        rows[0]["review_evidence" if changed_file == "manifest.jsonl" else "scene"] = (
            "changed" if changed_file == "manifest.jsonl" else {**rows[0]["scene"], "review_evidence": "changed"})
        write_jsonl(target, rows)
        with pytest.raises(DatasetError, match="changed after review opened"):
            review.export(base / "unpublished")
        assert not (base / "unpublished").exists()


def test_failed_final_audit_write_never_publishes_split_eligible_directory():
    with TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, journal = fixture_dataset(base)
        review = SceneReview(root, journal)
        review.append("S1", "confirm", "village", "实际用户")
        original_write = Path.write_text

        def fail_final_audit(path: Path, data: str, *args, **kwargs):
            if path.name == "review-audit.json":
                raise OSError("simulated final audit write failure")
            return original_write(path, data, *args, **kwargs)

        with patch.object(Path, "write_text", fail_final_audit):
            with pytest.raises(OSError, match="simulated"):
                review.export(base / "unpublished")
        assert not (base / "unpublished").exists()
        assert not list(base.glob(".unpublished.partial-*"))


def test_verdict_validation_and_source_path_escape():
    with TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, journal = fixture_dataset(base)
        review = SceneReview(root, journal)
        with pytest.raises(DatasetError, match="Reviewer"):
            review.append("S1", "confirm", "village", "")
        with pytest.raises(DatasetError, match="different"):
            review.append("S1", "correct", "village", "user")
        with pytest.raises(DatasetError, match="exact"):
            review.append("S1", "confirm", "battle", "user")
        row = read_jsonl(root / "manifest.jsonl")[0]
        row["image_path"] = "../other.png"
        write_jsonl(root / "manifest.jsonl", [row])
        with pytest.raises(DatasetError, match="escapes"):
            SceneReview(root, journal)


def test_http_requires_loopback_host_token_origin_and_valid_body():
    with TemporaryDirectory() as temporary:
        base = Path(temporary)
        root, journal = fixture_dataset(base)
        review = SceneReview(root, journal)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        token = "test-secret-token"
        server = HTTPServer(("127.0.0.1", port), make_handler(review, token, port))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base_url = f"http://127.0.0.1:{port}"

        def response(path: str, *, method: str = "GET", body: bytes | None = None,
                     headers: dict | None = None) -> int:
            request = Request(base_url + path, data=body, method=method, headers=headers or {})
            try:
                with urlopen(request, timeout=3) as result:
                    return result.status
            except HTTPError as error:
                return error.code

        try:
            assert response("/api/state") == 403
            assert response("/api/state?token=" + token) == 200
            assert response("/image/../secret?token=" + token) == 404
            assert response("/image/S1?token=" + token) == 200
            body = json.dumps({"sample_id": "S1", "action": "confirm", "label": "village", "reviewer": "实际用户"}).encode()
            valid_headers = {"Content-Type": "application/json", "Origin": base_url,
                             "X-Review-Token": token}
            assert response("/api/verdict", method="POST", body=body,
                            headers={**valid_headers, "Origin": "http://evil.invalid"}) == 403
            assert response("/api/verdict", method="POST", body=b"x" * 4097,
                            headers=valid_headers) == 413
            assert response("/api/verdict", method="POST", body=body, headers=valid_headers) == 200
            assert len(read_jsonl(journal)) == 1
            assert response("/image/no-such-id?token=" + token) == 404
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=3)
