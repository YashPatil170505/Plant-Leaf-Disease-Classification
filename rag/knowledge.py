"""
knowledge.py - Parse knowledge_base/*.md into retrievable chunks

Each KB file has YAML-like frontmatter (class, crop, disease, pathogen, sources)
followed by '## Section' blocks. Every section becomes one chunk that carries
its class name and the sources it cites.
"""

import os
import re
import glob

KB_DIR = "knowledge_base"
GENERAL_CLASS = "General___healthy_leaf_and_IPM"
CITE_TAG = re.compile(r"\[(S\d+)\]")


def _parse_frontmatter(text):
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    if not m:
        raise ValueError("missing frontmatter")
    meta, sources, current = {}, [], None
    for line in m.group(1).splitlines():
        if not line.strip() or line.strip() == "sources:":
            continue
        item = re.match(r"^\s*-\s*id:\s*(\S+)", line)
        field = re.match(r"^\s+(\w+):\s*(.*)$", line)
        top = re.match(r"^(\w+):\s*(.*)$", line)
        if item:
            current = {"id": item.group(1)}
            sources.append(current)
        elif field and current is not None:
            current[field.group(1)] = field.group(2).strip()
        elif top:
            meta[top.group(1)] = top.group(2).strip()
    meta["sources"] = sources
    return meta, text[m.end():]


def slug(section):
    return re.sub(r"[^a-z0-9]+", "_", section.lower()).strip("_")


def load_chunks(kb_dir=KB_DIR):
    chunks = []
    for path in sorted(glob.glob(os.path.join(kb_dir, "*.md"))):
        with open(path, encoding="utf-8") as f:
            meta, body = _parse_frontmatter(f.read())
        src_by_id = {s["id"]: s for s in meta["sources"]}

        for sec in re.split(r"^##\s+", body, flags=re.M)[1:]:
            title, _, text = sec.partition("\n")
            title, text = title.strip(), " ".join(text.split())
            if not text:
                continue
            cited = sorted(set(CITE_TAG.findall(text)), key=lambda s: int(s[1:]))
            chunks.append({
                "id": f"{meta['class']}#{slug(title)}",
                "class_name": meta["class"],
                "crop": meta.get("crop", ""),
                "disease": meta.get("disease", ""),
                "section": title,
                "text": text,
                "sources": [src_by_id[s] for s in cited if s in src_by_id],
            })
    return chunks


def embedding_text(chunk):
    """Text that gets embedded: header gives global search the crop/disease context."""
    return (f"Crop: {chunk['crop']}. Disease: {chunk['disease']}. "
            f"Section: {chunk['section']}. {CITE_TAG.sub('', chunk['text'])}")
