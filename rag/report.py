"""
report.py - Build a structured diagnosis report locally (no LLM involved)

The report is assembled from the classifier output and knowledge-base sections,
so every statement in it is traceable to the model or to a cited source.
"""

import os
import re
from datetime import datetime

from rag.knowledge import CITE_TAG

REPORT_SECTIONS = ["Symptoms", "Prevention and cultural control", "Chemical and biological control"]
DISCLAIMER = ("This report is generated automatically from an image classifier trained on lab photos "
              "(PlantVillage) and a curated reference set. It is not a substitute for diagnosis by a "
              "local agricultural extension officer or plant diagnostic lab. Always follow product "
              "labels and local regulations before applying any treatment.")


def _with_numbered_refs(text, chunk_sources, refs):
    """Rewrite per-file [S1] tags into report-wide [n] references."""
    by_id = {s["id"]: s for s in chunk_sources}

    def repl(m):
        src = by_id.get(m.group(1))
        if not src:
            return ""
        if src["url"] not in refs:
            refs[src["url"]] = (len(refs) + 1, src)
        return f"[{refs[src['url']][0]}]"
    return CITE_TAG.sub(repl, text)


def build_report(prediction, gating, retriever, out_dir="outputs"):
    top = prediction["top_k"][0]
    refs, lines = {}, []
    lines += [f"# Plant Leaf Diagnosis Report",
              f"",
              f"Generated: {datetime.now():%Y-%m-%d %H:%M}  ",
              f"Image: `{prediction['image']}`",
              f"",
              f"## Classifier result",
              f"",
              f"Status: **{gating['status'].upper()}** (confidence threshold {gating['threshold']:.0%})",
              f"",
              f"| Rank | Plant | Condition | Confidence |",
              f"|---|---|---|---|"]
    for i, c in enumerate(prediction["top_k"], 1):
        lines.append(f"| {i} | {c['plant']} | {c['disease']} | {c['confidence']:.1%} |")
    lines += ["", gating["message"], ""]

    # Disease info for the top class (or each candidate if uncertain)
    plausible = set(gating["filter_classes"])
    candidates = [top] if gating["status"] != "uncertain" else \
        [c for c in prediction["top_k"] if c["class_name"] in plausible]
    for c in candidates:
        kb_class = retriever.resolve_classes([c["class_name"]])
        chunks = [ch for ch in retriever.chunks if ch["class_name"] in kb_class]
        if not chunks:
            continue
        lines.append(f"## {c['plant']} - {c['disease']}")
        lines.append("")
        for sec in REPORT_SECTIONS:
            ch = next((x for x in chunks if x["section"] == sec), None)
            if ch:
                lines += [f"### {sec}", "", _with_numbered_refs(ch["text"], ch["sources"], refs), ""]

    lines += ["## References", ""]
    for url, (n, src) in sorted(refs.items(), key=lambda kv: kv[1][0]):
        lines.append(f"{n}. {src.get('title', '')} - {src.get('publisher', '')}. {url}")
    lines += ["", "## Disclaimer", "", DISCLAIMER, ""]

    os.makedirs(out_dir, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(os.path.basename(prediction["image"]))[0])
    path = os.path.join(out_dir, f"diagnosis_report_{stem}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path, len(refs)
