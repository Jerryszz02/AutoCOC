"""Source-bound U box review journal and loopback-only review UI."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import secrets
from urllib.parse import parse_qs, unquote, urlsplit

from autococ.unit_catalog import UNITS, get_unit
from .data import DatasetError, sha256
from .u_candidates import CandidateError, _json_unique, load_candidate_queue

MAX_BODY = 65536
HEX = frozenset("0123456789abcdef")


def _bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _jsonl(path):
    rows = []
    try:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                row = json.loads(line, object_pairs_hook=_json_unique)
                if not isinstance(row, dict):
                    raise DatasetError(f"Review line {n} is not an object")
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Cannot read U review journal: {exc}") from exc
    return rows


def _write_new(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _reviewer(value):
    if (not isinstance(value, str) or not value.strip() or len(value) > 80
            or any(ord(c) < 32 for c in value)):
        raise DatasetError("Reviewer must be explicit (1-80 printable characters)")
    return value.strip()


def _known_label(label):
    if label == "troop_group":
        return True
    if not isinstance(label, str) or not label.startswith("hero:"):
        return False
    unit = get_unit(label[5:])
    return unit is not None and unit.kind == "hero"


def _boxes(boxes, roi):
    if not isinstance(roi, list) or len(roi) != 4 or any(type(v) is not int for v in roi):
        raise DatasetError("Candidate ROI is not a fixed source-pixel rectangle")
    if not isinstance(boxes, list):
        raise DatasetError("Boxes must be a list")
    result = []
    for item in boxes:
        if not isinstance(item, dict) or set(item) != {"label", "bbox"}:
            raise DatasetError("Each U box must contain label and bbox")
        box = item["bbox"]
        if (not _known_label(item["label"]) or not isinstance(box, list) or len(box) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
                or not roi[0] <= box[0] < box[2] <= roi[2]
                or not roi[1] <= box[1] < box[3] <= roi[3]):
            raise DatasetError("U box identity or source-pixel bounds are invalid")
        result.append({"label": item["label"], "bbox": list(box)})
    return result


def authorize_local_request(*, host, origin, token, expected_token, port, require_origin):
    return (host == f"127.0.0.1:{port}" and token == expected_token
            and (not require_origin or origin == f"http://127.0.0.1:{port}"))


class UReviewJournal:
    """Append-only decisions bound to one queue and exact source-frame hashes."""

    def __init__(self, queue_path: Path, journal_path: Path, *, prototype_only: bool):
        if type(prototype_only) is not bool:
            raise DatasetError("Explicit prototype_only policy is required")
        self.queue_path = Path(queue_path).resolve(strict=True)
        try:
            self.queue, self.source_pool, self.queue_sha256 = load_candidate_queue(self.queue_path)
        except CandidateError as exc:
            raise DatasetError(str(exc)) from exc
        self.prototype_only = prototype_only
        self.journal_path = Path(journal_path).resolve()
        if self.journal_path.is_relative_to(self.source_pool):
            raise DatasetError("Journal must be outside immutable source pool")
        self.frames = {r["queue_frame_id"]: r for r in self.queue["frames"]}
        self.events = self._read_events()

    def _binding(self, frame_id):
        row = self.frames[frame_id]
        return {"queue_sha256": self.queue_sha256, "queue_frame_id": frame_id,
                "source_image_sha256": row["source_image_sha256"],
                "event_group_id": row["event_group_id"], "split": row["split"],
                "plan_sha256": row["plan_sha256"],
                "capture_manifest_sha256": row["capture_manifest_sha256"]}

    def _candidate(self, frame_id):
        if not isinstance(frame_id, str) or frame_id not in self.frames:
            raise DatasetError("Unknown U candidate frame")
        row = self.frames[frame_id]
        path = (self.source_pool / row["image"]).resolve(strict=True)
        if (not path.is_relative_to(self.source_pool) or path.is_symlink()
                or not path.is_file() or sha256(path) != row["sha256"]):
            raise DatasetError("Candidate source image changed")
        return row

    def _check_event(self, event, number, previous):
        if (type(event.get("schema_version")) is not int or event.get("schema_version") != 1
                or type(event.get("sequence")) is not int or event.get("sequence") != number
                or event.get("previous_sha256") != previous):
            raise DatasetError(f"Review journal chain invalid at row {number}")
        frame_id = event.get("queue_frame_id")
        if (not isinstance(frame_id, str) or frame_id not in self.frames
                or any(event.get(k) != v for k, v in self._binding(frame_id).items())):
            raise DatasetError(f"Review journal source binding invalid at row {number}")
        if event.get("review_method") not in {"human", "agent_visual"}:
            raise DatasetError(f"Review method missing at row {number}")
        if event["review_method"] == "agent_visual" and not self.prototype_only:
            raise DatasetError("Agent visual review is prototype-only")
        _reviewer(event.get("reviewer"))
        candidate = self.frames[frame_id]
        if (event.get("source_frame_id") != candidate["frame_id"]
                or event.get("source_frame_sha256") != candidate["source_image_sha256"]
                or event.get("source_roi_bbox") != candidate.get("roi_bbox")):
            raise DatasetError("Review event source frame/ROI mismatch")
        try:
            reviewed_at = datetime.fromisoformat(event.get("reviewed_at_utc", "").replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise DatasetError("Review event timestamp is invalid") from exc
        if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
            raise DatasetError("Review event timestamp must include a timezone")
        if (not isinstance(event.get("review_evidence"), str)
                or not event["review_evidence"].strip()
                or len(event["review_evidence"]) > 1000):
            raise DatasetError("Review evidence is missing or too long")
        if event.get("decision") == "complete":
            if candidate.get("import_blockers") or event.get("boxes_complete") is not True or event.get("exhaustive") is not True:
                raise DatasetError("Complete review lacks metadata or exhaustive confirmation")
            if event.get("quarantine_reason") is not None or event.get("draft_boxes") != []:
                raise DatasetError("Complete review cannot include quarantine reason")
            _boxes(event.get("boxes"), candidate["roi_bbox"])
        elif event.get("decision") == "quarantine":
            reason = event.get("quarantine_reason")
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 240:
                raise DatasetError("Quarantine requires a reason")
            if (event.get("boxes_complete") is not False or event.get("exhaustive") is not False
                    or event.get("boxes") != []):
                raise DatasetError("Quarantine cannot claim completeness")
            if candidate.get("roi_bbox") is not None:
                _boxes(event.get("draft_boxes", []), candidate["roi_bbox"])
            elif event.get("draft_boxes", []):
                raise DatasetError("Cannot keep draft boxes without a verified ROI")
        else:
            raise DatasetError("Unsupported review decision")
        digest = event.get("event_sha256")
        if (not isinstance(digest, str) or len(digest) != 64 or set(digest) - HEX
                or digest != _digest({k: v for k, v in event.items() if k != "event_sha256"})):
            raise DatasetError(f"Review event hash invalid at row {number}")

    def _read_events(self):
        if not self.journal_path.exists():
            return []
        rows, previous = _jsonl(self.journal_path), None
        for n, event in enumerate(rows, 1):
            self._check_event(event, n, previous)
            previous = event["event_sha256"]
        return rows

    def _append(self, frame_id, *, decision, boxes, exhaustive, reviewer,
                method, evidence, quarantine_reason=None):
        candidate = self._candidate(frame_id)
        reviewer = _reviewer(reviewer)
        if method not in {"human", "agent_visual"} or (method == "agent_visual" and not self.prototype_only):
            raise DatasetError("Review method violates selected policy")
        if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 1000:
            raise DatasetError("Review evidence must be explicit")
        if decision == "complete":
            boxes = _boxes(boxes, candidate["roi_bbox"])
            if candidate.get("import_blockers"):
                raise DatasetError("Candidate metadata is incomplete; quarantine it")
            if exhaustive is not True:
                raise DatasetError("Complete requires exhaustive ROI review")
            quarantine_reason = None
        elif decision == "quarantine":
            if not isinstance(quarantine_reason, str) or not quarantine_reason.strip():
                raise DatasetError("Quarantine requires a reason")
            roi = candidate.get("roi_bbox")
            boxes = _boxes(boxes, roi) if roi is not None else []
            exhaustive = False
        else:
            raise DatasetError("Unknown decision")
        self.events = self._read_events()
        event = {
            "schema_version": 1, "sequence": len(self.events) + 1,
            "previous_sha256": self.events[-1]["event_sha256"] if self.events else None,
            **self._binding(frame_id),
            "source_frame_id": candidate["frame_id"],
            "source_frame_sha256": candidate["source_image_sha256"],
            "source_roi_bbox": candidate.get("roi_bbox"),
            "decision": decision, "boxes": boxes if decision == "complete" else [],
            "draft_boxes": boxes if decision == "quarantine" else [],
            "boxes_complete": decision == "complete", "exhaustive": bool(exhaustive),
            "quarantine_reason": quarantine_reason, "reviewer": reviewer,
            "review_method": method, "review_evidence": evidence.strip(),
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        event["event_sha256"] = _digest(event)
        if not self.journal_path.exists():
            _write_new(self.journal_path, b"")
        with self.journal_path.open("ab") as stream:
            stream.write(_bytes(event) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(event)
        return event

    def append_human(self, frame_id, *, decision, boxes=None, exhaustive=False,
                     reviewer, quarantine_reason=None):
        return self._append(frame_id, decision=decision, boxes=boxes or [],
            exhaustive=exhaustive, reviewer=reviewer, method="human",
            evidence="Explicit local visual box review; no agent label was promoted.",
            quarantine_reason=quarantine_reason)

    def append_agent_visual(self, frame_id, *, decision, boxes=None, exhaustive=False,
                            reviewer, evidence, quarantine_reason=None):
        if not self.prototype_only:
            raise DatasetError("Agent visual review is prototype-only")
        return self._append(frame_id, decision=decision, boxes=boxes or [],
            exhaustive=exhaustive, reviewer=reviewer, method="agent_visual",
            evidence=evidence, quarantine_reason=quarantine_reason)

    def state(self):
        latest = {e["queue_frame_id"]: e for e in self.events}
        return {
            "queue_sha256": self.queue_sha256, "prototype_only": self.prototype_only,
            "hero_labels": sorted(f"hero:{u.unit_id}" for u in UNITS if u.kind == "hero"),
            "frames": [{
                "queue_frame_id": r["queue_frame_id"], "frame_id": r["frame_id"],
                "width": r["width"], "height": r["height"], "roi_bbox": r["roi_bbox"],
                "split": r["split"], "event_group_id": r["event_group_id"],
                "client_version": r["client_version"], "camera_stable": r["camera_stable"],
                "zoom_verified": r["zoom_verified"], "import_blockers": r["import_blockers"],
                "latest": ({"decision": latest[r["queue_frame_id"]]["decision"],
                            "reviewer": latest[r["queue_frame_id"]]["reviewer"],
                            "review_method": latest[r["queue_frame_id"]]["review_method"],
                            "boxes": latest[r["queue_frame_id"]]["boxes"],
                            "draft_boxes": latest[r["queue_frame_id"]]["draft_boxes"],
                            "exhaustive": latest[r["queue_frame_id"]]["exhaustive"],
                            "quarantine_reason": latest[r["queue_frame_id"]]["quarantine_reason"]}
                           if r["queue_frame_id"] in latest else None)
            } for r in self.queue["frames"]],
        }

    def review_rows(self):
        self.events = self._read_events()
        latest = {e["queue_frame_id"]: e for e in self.events}
        if set(self.frames) - set(latest):
            raise DatasetError("Every candidate requires an explicit complete or quarantine decision")
        journal_sha = sha256(self.journal_path)
        return [{
            "schema_version": 1, "queue_sha256": self.queue_sha256,
            "review_journal_sha256": journal_sha,
            "journal_event_sha256": e["event_sha256"],
            "queue_frame_id": frame_id, "decision": e["decision"],
            "review_method": e["review_method"], "reviewer": e["reviewer"],
            "review_evidence": e["review_evidence"],
            "source_frame_sha256": e["source_frame_sha256"],
            "source_roi_bbox": e["source_roi_bbox"],
            "boxes_complete": e["boxes_complete"], "exhaustive": e["exhaustive"],
            "quarantine_reason": e["quarantine_reason"], "boxes": e["boxes"],
        } for frame_id, e in ((fid, latest[fid]) for fid in self.frames)]

    def export_reviews(self, output: Path):
        output = Path(output).resolve()
        if output.exists():
            raise DatasetError(f"Refusing to overwrite review export: {output}")
        if output.is_relative_to(self.source_pool) or self.source_pool.is_relative_to(output):
            raise DatasetError("Review export must be separate from source pool")
        rows = self.review_rows()
        _write_new(output, b"".join(_bytes(row) + b"\n" for row in rows))
        return {"path": str(output), "sha256": sha256(output), "rows": len(rows),
                "complete": sum(r["decision"] == "complete" for r in rows),
                "quarantined": sum(r["decision"] == "quarantine" for r in rows)}


PAGE = """<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>U box review</title>
<style>body{font:15px system-ui;margin:0;background:#17202b;color:#f4f6fa}main{max-width:1200px;margin:auto;padding:18px}.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.panel{background:#263547;padding:12px;margin:10px 0}.wrap{position:relative;display:inline-block;max-width:100%}.wrap img{display:block;max-width:100%;height:auto}.wrap canvas{position:absolute;inset:0;width:100%;height:100%;touch-action:none}.warn{color:#ffd273}</style>
<main><h2>U-position box review</h2><p class="warn">Boxes use original pixels. If any visible target, identity, ROI, or completeness is uncertain, quarantine. Empty boxes mean a negative only after exhaustive review.</p>
<div class="bar"><button id="prev">Previous</button><span id="pos"></span><button id="next">Next</button><label>Reviewer <input id="reviewer" maxlength="80"></label></div><div id="meta" class="panel"></div>
<div class="wrap"><img id="img" alt="Original capture"><canvas id="cv"></canvas></div>
<div class="panel"><select id="label"></select><button id="undo">Undo</button><button id="clear">Clear</button><p><label><input type="checkbox" id="all"> I reviewed the full ROI and all visible U targets are labeled; none is unresolved.</label></p>
<input id="why" placeholder="Quarantine reason"><button id="complete">Complete</button><button id="quarantine">Quarantine</button><p id="status"></p></div></main>
<script>
const token=__TOKEN__;let d=null,i=0,boxes=[],drag=null;const $=x=>document.getElementById(x),q=encodeURIComponent;
async function req(url,opt){const r=await fetch(url,opt);let x={};try{x=await r.json()}catch{}if(!r.ok)throw Error(x.error||'Rejected');return x}
function pt(e){const c=$('cv'),r=c.getBoundingClientRect();return [(e.clientX-r.left)*c.width/r.width,(e.clientY-r.top)*c.height/r.height]}
function draw(){if(!d)return;const c=$('cv'),x=c.getContext('2d'),r=d.frames[i].roi_bbox;x.clearRect(0,0,c.width,c.height);if(r){x.strokeStyle='#ffdd62';x.lineWidth=Math.max(2,c.width/900);x.strokeRect(r[0],r[1],r[2]-r[0],r[3]-r[1])}for(const b of boxes){x.strokeStyle='#37f0a4';x.lineWidth=Math.max(2,c.width/700);x.strokeRect(...[b.bbox[0],b.bbox[1],b.bbox[2]-b.bbox[0],b.bbox[3]-b.bbox[1]])}if(drag){x.strokeStyle='#ff9951';x.setLineDash([6,5]);x.strokeRect(drag[0],drag[1],drag[2]-drag[0],drag[3]-drag[1]);x.setLineDash([])}}
function render(){const f=d.frames[i],latest=f.latest;$('pos').textContent=(i+1)+'/'+d.frames.length+' | '+f.split+' | '+f.event_group_id;boxes=latest?(latest.decision==='complete'?latest.boxes:(latest.draft_boxes||[])):[];$('all').checked=Boolean(latest&&latest.decision==='complete'&&latest.exhaustive);$('why').value=latest&&latest.decision==='quarantine'?latest.quarantine_reason:'';$('status').textContent='';$('complete').disabled=true;$('img').onerror=()=>{$('complete').disabled=true;$('status').textContent='Original image failed to load; complete review is disabled'};$('img').onload=()=>{const c=$('cv');c.width=$('img').naturalWidth;c.height=$('img').naturalHeight;const matches=c.width===f.width&&c.height===f.height;$('complete').disabled=Boolean(f.import_blockers.length||!matches);if(!matches)$('status').textContent='Image dimensions mismatch; quarantine only';draw()};$('img').src='/image/'+q(f.queue_frame_id)+'?token='+q(token);$('meta').textContent=JSON.stringify({frame:f.frame_id,version:f.client_version,roi:f.roi_bbox,camera_stable:f.camera_stable,zoom_verified:f.zoom_verified,blockers:f.import_blockers,latest:latest});draw()}
$('cv').onpointerdown=e=>{const p=pt(e);drag=[p[0],p[1],p[0],p[1]];$('cv').setPointerCapture(e.pointerId);draw()};
$('cv').onpointermove=e=>{if(drag){const p=pt(e);drag[2]=p[0];drag[3]=p[1];draw()}};
$('cv').onpointerup=e=>{if(!drag)return;const p=pt(e);drag[2]=p[0];drag[3]=p[1];const a=drag;drag=null;const f=d.frames[i],r=f.roi_bbox,b=[Math.min(a[0],a[2]),Math.min(a[1],a[3]),Math.max(a[0],a[2]),Math.max(a[1],a[3])];if(r&&b[0]>=r[0]&&b[1]>=r[1]&&b[2]<=r[2]&&b[3]<=r[3]&&b[2]>b[0]&&b[3]>b[1])boxes.push({label:$('label').value,bbox:b});draw()};
async function save(decision){const f=d.frames[i],reviewer=$('reviewer').value.trim();if(!reviewer){$('status').textContent='Reviewer required';return}const payload={queue_frame_id:f.queue_frame_id,decision:decision,reviewer:reviewer};if(decision==='complete'){payload.boxes=boxes;payload.exhaustive=$('all').checked;if(!payload.exhaustive){$('status').textContent='Exhaustive review required';return}}else{payload.draft_boxes=boxes;payload.quarantine_reason=$('why').value.trim();if(!payload.quarantine_reason){$('status').textContent='Quarantine reason required';return}}try{await req('/api/review',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:JSON.stringify(payload)});d=await req('/api/state?token='+q(token));render();$('status').textContent='Saved'}catch(e){$('status').textContent=e.message}}
$('complete').onclick=()=>save('complete');$('quarantine').onclick=()=>save('quarantine');$('undo').onclick=()=>{boxes.pop();draw()};$('clear').onclick=()=>{boxes=[];draw()};$('prev').onclick=()=>{i=Math.max(0,i-1);render()};$('next').onclick=()=>{i=Math.min(d.frames.length-1,i+1);render()};
(async()=>{try{d=await req('/api/state?token='+q(token));$('label').replaceChildren(...['troop_group',...d.hero_labels].map(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;return o}));render()}catch(e){$('status').textContent=e.message}})();
</script></html>"""


def make_handler(review, token, port):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status, body, mime):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Content-Security-Policy",
                "default-src 'none'; img-src 'self'; connect-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self, post):
            query = parse_qs(urlsplit(self.path).query)
            supplied = self.headers.get("X-Review-Token", "") if post else query.get("token", [""])[0]
            return authorize_local_request(host=self.headers.get("Host", ""),
                origin=self.headers.get("Origin"), token=supplied,
                expected_token=token, port=port, require_origin=post)

        def do_GET(self):
            if not self._authorized(False):
                self._reply(403, b'{"error":"forbidden"}', "application/json")
                return
            path = urlsplit(self.path).path
            if path == "/":
                self._reply(200, PAGE.replace("__TOKEN__", json.dumps(token)).encode(),
                            "text/html; charset=utf-8")
            elif path == "/api/state":
                self._reply(200, _bytes(review.state()), "application/json; charset=utf-8")
            elif path.startswith("/image/"):
                candidate = review.frames.get(unquote(path[len("/image/"):]))
                if candidate is None:
                    self._reply(404, b'{"error":"not found"}', "application/json")
                    return
                try:
                    image = (review.source_pool / candidate["image"]).resolve(strict=True)
                    if not image.is_relative_to(review.source_pool) or image.is_symlink():
                        raise DatasetError("Image path escaped source pool")
                    pixels = image.read_bytes()
                    if hashlib.sha256(pixels).hexdigest() != candidate["sha256"]:
                        raise DatasetError("Source image changed")
                    self._reply(200, pixels, "image/png")
                except (OSError, DatasetError):
                    self._reply(409, b'{"error":"source changed"}', "application/json")
            else:
                self._reply(404, b'{"error":"not found"}', "application/json")

        def do_POST(self):
            if not self._authorized(True) or urlsplit(self.path).path != "/api/review":
                self._reply(403, b'{"error":"forbidden"}', "application/json")
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                length = -1
            if not 1 <= length <= MAX_BODY:
                self._reply(413, b'{"error":"invalid body size"}', "application/json")
                return
            try:
                body = self.rfile.read(length)
                if len(body) != length:
                    raise DatasetError("Truncated request")
                payload = json.loads(body.decode("utf-8"), object_pairs_hook=_json_unique)
                if not isinstance(payload, dict):
                    raise DatasetError("Review body must be an object")
                if payload.get("decision") == "complete":
                    event = review.append_human(payload.get("queue_frame_id"),
                        decision="complete", boxes=payload.get("boxes"),
                        exhaustive=payload.get("exhaustive") is True,
                        reviewer=payload.get("reviewer"))
                elif payload.get("decision") == "quarantine":
                    event = review.append_human(payload.get("queue_frame_id"),
                        decision="quarantine", boxes=payload.get("draft_boxes", []),
                        reviewer=payload.get("reviewer"),
                        quarantine_reason=payload.get("quarantine_reason"))
                else:
                    raise DatasetError("Unsupported decision")
                self._reply(200, _bytes({"event_sha256": event["event_sha256"]}),
                            "application/json; charset=utf-8")
            except (UnicodeError, json.JSONDecodeError, DatasetError, CandidateError, OSError) as exc:
                self._reply(400, _bytes({"error": str(exc)[:240]}),
                            "application/json; charset=utf-8")

        def log_message(self, format, *args):
            return
    return Handler


def serve(review, port=8765):
    if type(port) is not int or not 1024 <= port <= 65535:
        raise DatasetError("Review port must be 1024-65535")
    token = secrets.token_urlsafe(32)
    server = HTTPServer(("127.0.0.1", port), make_handler(review, token, port))
    print(f"Open http://127.0.0.1:{port}/?token={token} (loopback only); Ctrl+C stops.")
    try:
        server.serve_forever()
    finally:
        server.server_close()
