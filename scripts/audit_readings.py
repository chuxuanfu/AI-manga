"""Offline diagnostic: preserve competing interpretations of the same illustration detail."""
import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "runtime/studio_pipeline"), str(ROOT / "extensions")]
from evidence import ask, backend, model_identity, objects, review_session
from print_judge import MODEL, fingerprint, store
from visual_triage import VERIFIER

READ_A = (
    'Describe the visible surfaces in this isolated illustration detail, in reading order. For each main '
    'surface give the most immediate visual reading and the specific marks that make it look that way. '
    'Mention any genuine competing reading. Do not infer how it continues beyond the frame. 65 words '
    'maximum. '
)
READ_B = (
    'Inventory the main subject in this illustrated region as for a continuity sheet. Name the visible '
    'surfaces from one edge to the other, including the physical construction of their endings, joins or '
    'attachments. Use specific nouns; distinguish clearly visible cues from inference. 55 words maximum. '
)
ALIGN = (
    'Build a complete component-by-component comparison of two independent captions of ONE image region. '
    'Start from B: for EACH distinct physical part and its extent, look for what A says occupies that '
    'same location. Then check remaining A parts. Do not stop at the first shared item. A nearby '
    'component is not the same component. Report all explicit contradictions about what visibly occupies '
    'the same space even if the overall object is agreed. Compatible detail, synonym or omission is not a '
    'contradiction. Quote exact short substrings; put empty string if absent. Limit to six rows; include '
    'major surfaces first. '
)
SELECT = (
    'You are editing a continuity sheet. Here are pairs of quotations about individual regions. Select '
    'only pairs that, taken literally, assert mutually exclusive visible identities, arrangement or '
    'extent for the SAME foreground subject part. Both claims must be specific. Two descriptions that can '
    'both be true, differing adjectives, a generic description and a more specific one, and unmentioned '
    'details all give no selection. Do not speculate beyond the text. Return selected index numbers; '
    'empty array if none. '
)
CHECK = (
    'Look specifically at the two quoted descriptions of the indicated component of this image. Inspect '
    'its outlines and how they join. Could both descriptions be true of the same component in this '
    'drawing? If yes, explain how and mark compatible. If one reading is clearly unsupported, mark '
    'resolved. If the image itself visibly invites two incompatible readings, give the local cue for '
    'each, mark ambiguous. Avoid choosing the intended answer by inventing detail. State only pixel '
    'evidence, within 50 words. '
)


def schema(fields):
    return {"type": "object", "properties": fields, "required": list(fields)}


STRING = {"type": "string"}
PART = schema({"part": STRING, "b": STRING, "a": STRING,
               "comparison": {"enum": ["incompatible", "consistent", "not_compared"]}})
PARTS = schema({"parts": {"type": "array", "items": PART}})
INDICES = schema({"indices": {"type": "array", "items": {"type": "integer"}}})
DECISION = schema({"reason": STRING, "decision": {"enum": ["compatible", "resolved", "ambiguous"]}})


def strips(rect):
    left, top, right, bottom = rect
    w, h = right - left, bottom - top
    for a, b in [(0, .39), (.30, .73), (.65, 1)]:
        if h >= w:
            yield (left, top + round(a*h), right, top + round(b*h))
        else:
            yield (left + round(a*w), top, left + round(b*w), bottom)


def audit(path):
    with Image.open(path) as source:
        image = source.convert("RGB")
    found = objects(ImageOps.contain(image, (1100, 1100)))
    report = {"source": str(path.resolve()), "sha256": fingerprint(path), "size": image.size,
              "models": {tag: model_identity(tag) for tag in [MODEL, VERIFIER]},
              "readings": [], "comparisons": [], "checks": []}
    if not found:
        return dict(report, status="no_detected_subject")

    def area(item):
        l, t, r, b = item["box"]
        return (r-l)*(b-t)

    subject = max(found, key=area)
    rect = tuple(round(v*n/1000) for v, n in zip(subject["box"], image.size*2))
    report["subject"] = subject
    rows = [{"pixels": box} for box in strips(rect)]
    report["readings"] = rows
    views = [ImageOps.contain(image.crop(row["pixels"]), (850, 850)) for row in rows]
    with review_session(), backend(VERIFIER) as url:
        for tag, prompt, label in [(VERIFIER, READ_A, "first"), (MODEL, READ_B, "second")]:
            for row, view in zip(rows, views):
                row[label] = ask(url, [view], prompt.strip(), tag=tag)
        pairs = report["comparisons"]
        for i, row in enumerate(rows):
            prompt = ALIGN.strip() + "\nCaption B: " + row["second"] + "\nCaption A: " + row["first"]
            aligned = json.loads(ask(url, [], prompt, PARTS, tag=MODEL))["parts"]
            for part in aligned:
                if all(part[k].strip() and part[k].casefold() in row[s].casefold()
                       for k, s in [("a", "first"), ("b", "second")]):
                    pairs.append({"strip": i, "part": part["part"], "a": part["a"], "b": part["b"]})
        if pairs:
            sheet = [{"i": i, "part": pair["part"], "A": pair["a"], "B": pair["b"]}
                     for i, pair in enumerate(pairs)]
            indices = json.loads(ask(url, [], SELECT.strip()+"\n"+json.dumps(sheet), INDICES, tag=VERIFIER))["indices"]
            if any(type(i) is not int or not 0 <= i < len(pairs) for i in indices):
                raise ValueError("Invalid comparison index returned by the local model")
            for i in dict.fromkeys(indices):
                pair = pairs[i]
                claim = {k: pair[k] for k in ["part", "a", "b"]}
                reply = ask(url, [views[pair["strip"]]], CHECK.strip()+"\n"+json.dumps(claim),
                            DECISION, tag=MODEL, reason=True)
                report["checks"].append(dict(pair=i, **json.loads(reply)))
    return dict(report, status="diagnostic_complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.suffix.lower() != ".json" or args.report.resolve() == args.image.resolve():
        parser.error("Choose a separate JSON report file.")
    os.environ["PATH"] = str(ROOT/"runtime/tools/bin") + os.pathsep + os.environ["PATH"]
    receipt = audit(args.image)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    store(args.report, receipt)
    print(args.report)
