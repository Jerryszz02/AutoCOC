"""Local, explicit human scene review over immutable source evidence.

This module never infers a human verdict from an agent annotation.  The journal
is separate from the source dataset and every verdict is bound to its exact
manifest, annotations, row, and image bytes.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import shutil
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, quote, urlsplit

from .data import DatasetError, image_path, load_taxonomy, read_jsonl, sha256, validate, write_jsonl


MAX_BODY = 4096
MAX_REVIEWER = 80
HEX64 = set("0123456789abcdef")


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _indexed(rows: list[dict], name: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in result:
            raise DatasetError(f"Duplicate/missing {name} sample_id: {sample_id!r}")
        result[sample_id] = row
    return result


class SceneReview:
    def __init__(self, root: Path, journal: Path, *, manifest: str = "manifest.jsonl",
                 annotations: str = "annotations.jsonl") -> None:
        self.root = Path(root).resolve(strict=True)
        self.manifest_path = image_path(self.root, manifest)
        self.annotations_path = image_path(self.root, annotations)
        self.journal = Path(journal).resolve()
        if self.journal.is_relative_to(self.root):
            raise DatasetError("Review journal must be outside the source dataset")
        self.manifest = read_jsonl(self.manifest_path)
        self.annotations = read_jsonl(self.annotations_path)
        checks = validate(self.root, self.manifest, self.annotations, verify_pixels=True)
        if not checks["valid"]:
            raise DatasetError("Source dataset invalid: " + "; ".join(checks["errors"][:5]))
        self.samples = _indexed(self.manifest, "manifest")
        self.labels = _indexed(self.annotations, "annotation")
        if set(self.samples) != set(self.labels):
            raise DatasetError("Every source sample needs an annotation row")
        self.classes = tuple(load_taxonomy()["scene_classes"] + [load_taxonomy()["unknown_scene"]])
        self.manifest_sha256 = sha256(self.manifest_path)
        self.annotations_sha256 = sha256(self.annotations_path)
        self.order = list(self.samples)
        for sample_id in self.order:
            item = self.labels[sample_id].get("scene")
            if not isinstance(item, dict) or item.get("label") not in self.classes:
                raise DatasetError(f"{sample_id}: missing scene suggestion")
            path = image_path(self.root, self.samples[sample_id]["image_path"])
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                raise DatasetError(f"{sample_id}: unsupported image format")
        self.events = self._read_journal()

    def _binding(self, sample_id: str) -> dict:
        row = self.samples[sample_id]
        return {"source_manifest_sha256": self.manifest_sha256,
                "source_annotations_sha256": self.annotations_sha256,
                "source_manifest_row_sha256": _digest(row),
                "source_annotation_row_sha256": _digest(self.labels[sample_id]),
                "source_image_sha256": row["file_sha256"]}

    def _check_source(self, sample_id: str) -> Path:
        row = self.samples[sample_id]
        path = image_path(self.root, row["image_path"])
        if not path.is_file() or sha256(path) != row["file_sha256"]:
            raise DatasetError(f"{sample_id}: source image SHA-256 mismatch")
        if sha256(self.manifest_path) != self.manifest_sha256 or sha256(self.annotations_path) != self.annotations_sha256:
            raise DatasetError("Source manifest/annotations changed after review opened")
        return path

    def _read_journal(self) -> list[dict]:
        if not self.journal.exists():
            return []
        events = read_jsonl(self.journal)
        previous = None
        for index, event in enumerate(events):
            sample_id = event.get("sample_id")
            if sample_id not in self.samples:
                raise DatasetError(f"Journal entry {index + 1}: unknown sample")
            if event.get("schema_version") != 1 or event.get("sequence") != index + 1 or event.get("previous_sha256") != previous:
                raise DatasetError(f"Journal entry {index + 1}: sequence/chain mismatch")
            event_hash = event.get("event_sha256")
            if not isinstance(event_hash, str) or len(event_hash) != 64 or set(event_hash) - HEX64:
                raise DatasetError(f"Journal entry {index + 1}: invalid event hash")
            if event_hash != _digest({key: value for key, value in event.items() if key != "event_sha256"}):
                raise DatasetError(f"Journal entry {index + 1}: event hash mismatch")
            if any(event.get(key) != value for key, value in self._binding(sample_id).items()):
                raise DatasetError(f"Journal entry {index + 1}: source binding mismatch")
            self._validate_verdict(sample_id, event["action"], event.get("label"), event.get("reviewer"))
            previous = event_hash
        return events

    def _validate_verdict(self, sample_id: str, action: str, label: str | None, reviewer: str) -> None:
        if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > MAX_REVIEWER or any(ord(c) < 32 for c in reviewer):
            raise DatasetError("Reviewer must be explicitly entered (1-80 printable characters)")
        suggestion = self.labels[sample_id]["scene"]["label"]
        if action == "confirm" and label != suggestion:
            raise DatasetError("Confirm must use the exact source suggestion")
        if action == "correct" and (label not in self.classes or label == suggestion):
            raise DatasetError("Correction must choose a different taxonomy label")
        if action == "skip" and label is not None:
            raise DatasetError("Skip cannot carry a label")
        if action not in {"confirm", "correct", "skip"}:
            raise DatasetError("Unknown review action")

    def append(self, sample_id: str, action: str, label: str | None, reviewer: str) -> dict:
        if sample_id not in self.samples:
            raise DatasetError("Unknown sample_id")
        self._validate_verdict(sample_id, action, label, reviewer)
        self._check_source(sample_id)
        # Re-read before append so external modifications cannot be silently lost.
        self.events = self._read_journal()
        event = {"schema_version": 1, "sequence": len(self.events) + 1,
                 "previous_sha256": self.events[-1]["event_sha256"] if self.events else None,
                 "sample_id": sample_id, "action": action, "label": label,
                 "reviewer": reviewer.strip(),
                 "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
                 **self._binding(sample_id)}
        event["event_sha256"] = _digest(event)
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.journal, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "ab") as stream:
            stream.write(_json_bytes(event) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(event)
        return event

    def latest(self) -> dict[str, dict]:
        return {event["sample_id"]: event for event in self.events}

    def state(self) -> dict:
        latest = self.latest()
        return {"source_manifest_sha256": self.manifest_sha256,
                "samples": [{"sample_id": sample_id,
                             "client_version": self.samples[sample_id].get("client_version"),
                             "source_session": self.samples[sample_id].get("source_session"),
                             "leakage_group_id": self.samples[sample_id].get("leakage_group_id"),
                             "source_ref": self.samples[sample_id].get("source_ref"),
                             "suggestion": self.labels[sample_id]["scene"]["label"],
                             "agent_reviewer": self.labels[sample_id]["scene"].get("reviewer"),
                             "latest": {key: latest[sample_id][key] for key in ("action", "label", "reviewer", "reviewed_at_utc")}
                             if sample_id in latest else None}
                            for sample_id in self.order],
                "classes": self.classes}

    def export(self, output: Path) -> dict:
        output = Path(output).resolve()
        if output.exists():
            raise DatasetError(f"Refusing to overwrite export: {output}")
        if output == self.root or output.is_relative_to(self.root) or self.root.is_relative_to(output):
            raise DatasetError("Export must be separate from source dataset")
        self.events = self._read_journal()
        latest = self.latest()
        approved = [sample_id for sample_id in self.order if sample_id in latest and latest[sample_id]["action"] in {"confirm", "correct"}]
        if not approved:
            raise DatasetError("No explicit confirmed scene verdicts to export")
        for sample_id in approved:
            self._check_source(sample_id)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Build in a sibling directory on the same filesystem. A failed final
        # audit write or validation must never leave a split-eligible output.
        with TemporaryDirectory(prefix=f".{output.name}.partial-", dir=output.parent) as temporary:
            staging = Path(temporary)
            audit = self._write_export(staging, approved, latest)
            # On Windows, rename fails when the destination already exists.
            # Check again to make the no-overwrite intent clear to callers.
            if output.exists():
                raise DatasetError(f"Refusing to overwrite export: {output}")
            staging.rename(output)
        return {"output": str(output), **audit}

    def _write_export(self, staging: Path, approved: list[str], latest: dict[str, dict]) -> dict:
        manifest, annotations = [], []
        for sample_id in approved:
            original, annotation, verdict = self.samples[sample_id], self.labels[sample_id], latest[sample_id]
            # Do not break a parent reference when exporting a subset.
            if original.get("parent_sample_id") and original["parent_sample_id"] not in approved:
                raise DatasetError(f"{sample_id}: parent sample must also be confirmed")
            source_image = self._check_source(sample_id)
            image = image_path(staging, original["image_path"])
            image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_image, image)
            if sha256(image) != original["file_sha256"]:
                raise DatasetError(f"{sample_id}: copied image SHA-256 mismatch")
            evidence = f"human-verdicts.jsonl event_sha256={verdict['event_sha256']}"
            manifest.append({**original, "review_status": "reviewed", "review_method": "human",
                             "reviewer": verdict["reviewer"], "review_evidence": evidence,
                             "annotation_version": "scene-human-review-v1",
                             "source_review": {key: original.get(key) for key in ("review_status", "review_method", "reviewer", "review_evidence")}})
            scene = {**annotation["scene"], "label": verdict["label"], "complete": True,
                     "review_status": "reviewed", "review_method": "human",
                     "reviewer": verdict["reviewer"], "review_evidence": evidence,
                     "source_agent_scene": annotation["scene"]}
            annotations.append({**annotation, "scene": scene})
        write_jsonl(staging / "manifest.jsonl", manifest)
        write_jsonl(staging / "annotations.jsonl", annotations)
        write_jsonl(staging / "human-verdicts.jsonl", self.events)
        write_jsonl(staging / "source-manifest.jsonl", self.manifest)
        write_jsonl(staging / "source-annotations.jsonl", self.annotations)
        report = validate(staging, manifest, annotations, verify_pixels=True)
        if not report["valid"]:
            raise DatasetError("Export validation failed: " + "; ".join(report["errors"][:5]))
        audit = {"schema_version": 1, "source_manifest_sha256": self.manifest_sha256,
                 "source_annotations_sha256": self.annotations_sha256,
                 "confirmed_samples": len(approved), "journal_events": len(self.events),
                 "policy": "explicit human scene review only; no split, training, or model release"}
        (staging / "review-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return audit


PAGE = """<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>AutoCOC 场景逐图复核</title>
<style>body{font:16px system-ui;margin:0;background:#161d29;color:#f3f5fa}main{max-width:1260px;margin:auto;padding:20px}h1{font-size:24px}.bar,.actions{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:12px 0}.card{background:#273245;padding:14px;border-radius:10px}img{display:block;width:100%;height:auto;max-height:66vh;object-fit:contain;background:#080b10}button,select,input{font:inherit;padding:8px;border-radius:6px}button{cursor:pointer}button:focus-visible{outline:3px solid #ffcf57}.warn{color:#ffcf57}#viewer{overflow:auto}#viewer.zoom img{width:200%;max-height:none;max-width:none}small{overflow-wrap:anywhere}</style>
<main><h1>场景逐图人工复核</h1><p class=\"warn\">请查看原图后，明确选择确认、改标签或跳过。建议来自 Agent，尚未成为人工真值。操作会保留历史；页面不会自动确认或跳到下一张。</p>
<div class=\"bar\"><button id=\"prev\">← 上一张</button><strong id=\"position\"></strong><button id=\"next\">下一张 →</button><button id=\"zoom\">放大 / 适应</button></div>
<div class=\"card\" id=\"meta\"></div><div id=\"viewer\"><img id=\"image\" alt=\"待复核原图\"></div>
<div class=\"card\"><div class=\"bar\"><label>复核人（本人填写） <input id=\"reviewer\" maxlength=\"80\" autocomplete=\"off\"></label><label>改为 <select id=\"label\"></select></label></div>
<div class=\"actions\"><button id=\"confirm\">确认建议</button><button id=\"correct\">确认改为所选标签</button><button id=\"skip\">跳过 / 撤回此图</button></div><p id=\"status\" role=\"status\"></p></div><p><small>方向键切换图片；Z 放大。为防误操作，快捷键不提交复核结果。</small></p></main>
<script>const token=__TOKEN__;let data=null,idx=0;const $=id=>document.getElementById(id);const q=v=>encodeURIComponent(v);
async function load(){let r=await fetch('/api/state?token='+q(token));if(!r.ok)throw Error('读取失败');data=await r.json();$('label').replaceChildren(...data.classes.map(c=>{let o=document.createElement('option');o.value=c;o.textContent=c;return o}));render()}
function render(){let s=data.samples[idx];$('position').textContent=`${idx+1} / ${data.samples.length}`;$('image').src='/image/'+q(s.sample_id)+'?token='+q(token);$('label').value=s.suggestion;let m=$('meta');m.replaceChildren();for(let [k,v] of Object.entries({样本:s.sample_id,客户端:s.client_version,来源会话:s.source_session,泄漏分组:s.leakage_group_id,来源:s.source_ref,Agent建议:s.suggestion,Agent复核人:s.agent_reviewer,最新人工操作:s.latest?`${s.latest.action} · ${s.latest.label||'无'} · ${s.latest.reviewer} · ${s.latest.reviewed_at_utc}`:'未复核'})){let p=document.createElement('p');p.textContent=k+'：'+(v??'未知');m.append(p)}$('status').textContent=''}
async function save(action){let s=data.samples[idx],reviewer=$('reviewer').value.trim();if(!reviewer){$('status').textContent='请本人填写复核人';return}let label=action==='confirm'?s.suggestion:action==='correct'?$('label').value:null;if(action==='correct'&&label===s.suggestion){$('status').textContent='改标签需选择不同类别';return}if(!window.confirm(`保存 ${s.sample_id}：${action}${label?' / '+label:''}？`))return;let r=await fetch('/api/verdict',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:JSON.stringify({sample_id:s.sample_id,action,label,reviewer})});let v=await r.json();$('status').textContent=r.ok?'已保存，历史保留；请手动切换到下一张':(v.error||'保存失败');if(r.ok){let fresh=await fetch('/api/state?token='+q(token));data=await fresh.json();render();$('status').textContent='已保存，历史保留；请手动切换到下一张'}}
$('prev').onclick=()=>{idx=Math.max(0,idx-1);render()};$('next').onclick=()=>{idx=Math.min(data.samples.length-1,idx+1);render()};$('zoom').onclick=()=>$('viewer').classList.toggle('zoom');for(let a of ['confirm','correct','skip'])$(a).onclick=()=>save(a);document.onkeydown=e=>{if(['INPUT','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowLeft')$('prev').click();if(e.key==='ArrowRight')$('next').click();if(e.key.toLowerCase()==='z')$('zoom').click()};load().catch(e=>$('status').textContent=e.message);
</script></html>"""


def make_handler(review: SceneReview, token: str, port: int) -> type[BaseHTTPRequestHandler]:
    origin = f"http://127.0.0.1:{port}"

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: bytes, mime: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Content-Security-Policy", "default-src 'none'; img-src 'self'; connect-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _allowed(self) -> bool:
            return self.headers.get("Host") == f"127.0.0.1:{port}"

        def do_GET(self) -> None:
            if not self._allowed():
                self._reply(403, b"Forbidden", "text/plain; charset=utf-8")
                return
            parsed = urlsplit(self.path)
            if parse_qs(parsed.query).get("token") != [token]:
                self._reply(403, b"Forbidden", "text/plain; charset=utf-8")
                return
            if parsed.path == "/":
                page = PAGE.replace("__TOKEN__", json.dumps(token)).encode("utf-8")
                self._reply(200, page, "text/html; charset=utf-8")
            elif parsed.path == "/api/state":
                self._reply(200, _json_bytes(review.state()), "application/json; charset=utf-8")
            elif parsed.path.startswith("/image/"):
                from urllib.parse import unquote
                sample_id = unquote(parsed.path[len("/image/"):])
                if sample_id not in review.samples:
                    self._reply(404, b"Not found", "text/plain; charset=utf-8")
                    return
                try:
                    path = review._check_source(sample_id)
                    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}[path.suffix.lower()]
                    pixels = path.read_bytes()
                    if hashlib.sha256(pixels).hexdigest() != review.samples[sample_id]["file_sha256"]:
                        raise DatasetError("Source image changed during read")
                    self._reply(200, pixels, mime)
                except (OSError, DatasetError):
                    self._reply(409, b"Source changed", "text/plain; charset=utf-8")
            else:
                self._reply(404, b"Not found", "text/plain; charset=utf-8")

        def do_POST(self) -> None:
            if (not self._allowed() or self.headers.get("Origin") != origin
                    or self.headers.get("X-Review-Token") != token):
                self._reply(403, _json_bytes({"error": "请求来源验证失败"}), "application/json; charset=utf-8")
                return
            if urlsplit(self.path).path != "/api/verdict" or self.headers.get("Content-Type") != "application/json":
                self._reply(404, _json_bytes({"error": "无效请求"}), "application/json; charset=utf-8")
                return
            length = self.headers.get("Content-Length", "")
            if not length.isdecimal() or not 0 < int(length) <= MAX_BODY:
                self._reply(413, _json_bytes({"error": "请求过大"}), "application/json; charset=utf-8")
                return
            try:
                payload = json.loads(self.rfile.read(int(length)))
                if not isinstance(payload, dict) or set(payload) != {"sample_id", "action", "label", "reviewer"}:
                    raise DatasetError("请求字段无效")
                event = review.append(payload["sample_id"], payload["action"], payload["label"], payload["reviewer"])
                self._reply(200, _json_bytes({"sequence": event["sequence"], "event_sha256": event["event_sha256"]}), "application/json; charset=utf-8")
            except (DatasetError, OSError, ValueError, TypeError, KeyError) as exc:
                self._reply(400, _json_bytes({"error": str(exc)}), "application/json; charset=utf-8")

        def log_message(self, format: str, *args: object) -> None:
            pass

    return Handler


def serve(review: SceneReview, port: int = 8765) -> None:
    if not 1 <= port <= 65535:
        raise DatasetError("Port must be 1-65535")
    token = secrets.token_urlsafe(32)
    with HTTPServer(("127.0.0.1", port), make_handler(review, token, port)) as server:
        print(f"请在本机浏览器打开 http://127.0.0.1:{port}/?token={quote(token)}", flush=True)
        print("Ctrl+C 停止。仅本人查看并提交逐图场景复核。", flush=True)
        server.serve_forever(poll_interval=0.5)
