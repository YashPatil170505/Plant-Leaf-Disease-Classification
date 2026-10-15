"""
agent.py - Plant disease assistant: Gemini for conversation, local tools for everything else

Tools (all executed locally, chosen by Gemini via function calling):
  predict_disease            - ResNet50 classifier on a leaf image
  retrieve_disease_info      - dense retrieval over the curated knowledge base
  generate_diagnosis_report  - structured Markdown report built from the above

Guardrails enforced in code (not just in the prompt):
  - Confidence gating: below the threshold the prediction is marked "uncertain"
    and retrieval covers all top-3 candidates instead of committing to one.
  - Retrieval is filtered to the predicted class(es) when a prediction exists.
  - Citations [P#] in the answer are checked against the passages actually
    retrieved; unknown ids are reported as invalid.
"""

import os
import re
import time
from google.genai import types

from rag import llm
from rag.retriever import Retriever
from rag.report import build_report

CONFIDENCE_THRESHOLD = 0.80     # from evaluation/confidence_threshold.py: flags 17/29 test errors, rejects ~1%
MIN_CANDIDATE_CONF   = 0.10     # uncertain case: only candidates at least this likely are discussed
TOP_K_PASSAGES       = 4
MAX_TOOL_ROUNDS      = 5
CITATION             = re.compile(r"P(\d+)")          # matches [P1], [P1, P4], [P1][P2]
CITE_GROUP           = re.compile(r"\[(P\d+(?:\s*,\s*P\d+)*)\]")

SYSTEM_PROMPT = """You are a plant disease assistant attached to a leaf-image classifier (ResNet50 trained on the PlantVillage dataset, 38 classes across 14 crops).

Tools:
- predict_disease(image_path): run the classifier. Call it whenever the user gives an image path. Never guess a diagnosis yourself.
- retrieve_disease_info(query, class_names?): search the reference library. Call it before stating ANY fact about symptoms, causes, conditions, prevention or treatment. class_names may be omitted when a prediction exists; the tool then filters to the predicted class(es) automatically.
- generate_diagnosis_report(): write a report file for the latest prediction. Call it only when the user asks for a report.

Rules:
1. Answer only from retrieved passages. Cite each factual sentence with the passage id in square brackets, e.g. [P2], or [P1][P3] for several. Only cite ids that appear in tool results.
2. If the passages do not contain the answer, say "My reference library doesn't cover that" and suggest asking a local agricultural extension officer. Do not fill gaps from general knowledge.
3. If a prediction has status "uncertain", say clearly that the model is unsure, list the top candidates with their confidence, and do not present any one of them as the diagnosis. Suggest a clearer photo of a single leaf on a plain background.
4. If the status is "healthy", say no disease was detected; do not invent problems.
5. Never give pesticide doses or mixing rates. When discussing chemical control, remind the user to follow the product label and local extension advice.
6. The classifier was trained on lab photos; mention that field photos may be less reliable when relevant.
7. Be concise: short paragraphs, plain language, no more than about 200 words unless the user asks for detail."""

TOOLS = [types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="predict_disease",
        description="Classify a plant leaf image with the local ResNet50 model. Returns top-3 classes with confidence and a gating status (confident / uncertain / healthy).",
        parameters_json_schema={
            "type": "object",
            "properties": {"image_path": {"type": "string", "description": "Path to the leaf image file"}},
            "required": ["image_path"],
        },
    ),
    types.FunctionDeclaration(
        name="retrieve_disease_info",
        description="Search the curated plant-disease reference library (university extension / government sources). Returns passages with ids like P1 to cite.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look up, e.g. 'how does late blight spread'"},
                "class_names": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Optional class names (e.g. 'Tomato___Late_blight') to restrict the search. Omit to use the latest prediction.",
                },
            },
            "required": ["query"],
        },
    ),
    types.FunctionDeclaration(
        name="generate_diagnosis_report",
        description="Generate a Markdown diagnosis report (prediction, symptoms, management, references) for the latest prediction and save it to disk.",
        parameters_json_schema={"type": "object", "properties": {}},
    ),
])]


def gate(prediction, threshold=CONFIDENCE_THRESHOLD):
    top = prediction["top_k"][0]
    if top["confidence"] < threshold:
        plausible = [c for c in prediction["top_k"] if c["confidence"] >= MIN_CANDIDATE_CONF] or [top]
        names = ", ".join(f"{c['plant']} {c['disease']} ({c['confidence']:.0%})" for c in plausible)
        return {"status": "uncertain", "threshold": threshold,
                "message": f"Low confidence ({top['confidence']:.0%} < {threshold:.0%}). "
                           f"Possible conditions: {names}. Do not treat any single one as the diagnosis.",
                "filter_classes": [c["class_name"] for c in plausible]}
    if top["is_healthy"]:
        return {"status": "healthy", "threshold": threshold,
                "message": f"No disease detected: {top['plant']} leaf classified as healthy ({top['confidence']:.0%}).",
                "filter_classes": [top["class_name"]]}
    return {"status": "confident", "threshold": threshold,
            "message": f"Predicted {top['plant']} - {top['disease']} ({top['confidence']:.0%}).",
            "filter_classes": [top["class_name"]]}


class PlantAssistant:
    def __init__(self, retriever=None, classifier=None, verbose=True):
        self.retriever  = retriever or Retriever()
        self._classifier = classifier
        self.verbose    = verbose
        self.contents   = []      # full Gemini conversation (append-only)
        self.passages   = {}      # "P1" -> chunk, accumulated over the session
        self.prediction = None
        self.gating     = None

    @property
    def classifier(self):
        if self._classifier is None:          # lazy: torch model only loads if an image is used
            from model import LeafClassifier
            self._classifier = LeafClassifier()
        return self._classifier

    # ── local tools ───────────────────────────────────────────────────
    def tool_predict_disease(self, image_path):
        if not os.path.isfile(image_path):
            return {"error": f"Image not found: {image_path}"}
        self.prediction = self.classifier.predict(image_path, topk=3)
        self.gating = gate(self.prediction)
        return {"top_k": self.prediction["top_k"],
                "status": self.gating["status"],
                "threshold": self.gating["threshold"],
                "note": self.gating["message"]}

    def tool_retrieve_disease_info(self, query, class_names=None):
        if not class_names and self.gating:
            class_names = self.gating["filter_classes"]
        hits = self.retriever.search(query, k=TOP_K_PASSAGES, class_filter=class_names or None)
        out = []
        for h in hits:
            pid = next((p for p, c in self.passages.items() if c["id"] == h["id"]), None)
            if pid is None:
                pid = f"P{len(self.passages) + 1}"
                self.passages[pid] = h
            out.append({"id": pid, "crop": h["crop"], "disease": h["disease"], "section": h["section"],
                        "text": re.sub(r"\s*\[S\d+\]", "", h["text"]),
                        "sources": [s.get("publisher", "") for s in h["sources"]]})
        return {"filtered_to": class_names or "all classes", "passages": out}

    def tool_generate_diagnosis_report(self):
        if not self.prediction:
            return {"error": "No prediction yet. Ask the user for a leaf image first."}
        path, n_refs = build_report(self.prediction, self.gating, self.retriever)
        return {"report_path": path, "status": self.gating["status"], "references": n_refs}

    def _run_tool(self, call):
        fn = getattr(self, f"tool_{call.name}", None)
        if fn is None:
            return {"error": f"Unknown tool {call.name}"}
        try:
            return fn(**(call.args or {}))
        except Exception as e:                  # report tool failures back to the model
            return {"error": f"{type(e).__name__}: {e}"}

    # ── conversation loop ─────────────────────────────────────────────
    def ask(self, user_text):
        """Run one user turn. Returns a dict with the answer, tool trace, citations and timings."""
        t_start = time.perf_counter()
        trace, llm_ms, tool_ms, model_used = [], 0.0, 0.0, None
        self.contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))

        for _ in range(MAX_TOOL_ROUNDS):
            t0 = time.perf_counter()
            resp, model_used = llm.chat_turn(self.contents, SYSTEM_PROMPT, TOOLS)
            llm_ms += (time.perf_counter() - t0) * 1000

            candidate = resp.candidates[0].content if resp.candidates else None
            if candidate is None:
                answer = "(The model returned no content.)"
                break
            self.contents.append(candidate)       # keep model turns verbatim (thought signatures)

            calls = resp.function_calls or []
            if not calls:
                answer = (resp.text or "").strip()
                break

            parts = []
            for call in calls:
                t0 = time.perf_counter()
                result = self._run_tool(call)
                tool_ms += (time.perf_counter() - t0) * 1000
                trace.append({"tool": call.name, "args": dict(call.args or {}), "result": result})
                if self.verbose:
                    print(f"  [tool] {call.name}({dict(call.args or {})})")
                parts.append(types.Part.from_function_response(name=call.name, response=result))
            self.contents.append(types.Content(role="user", parts=parts))
        else:
            answer = "(Stopped: too many tool calls in one turn.)"

        groups  = " ".join(CITE_GROUP.findall(answer))
        cited   = sorted({f"P{n}" for n in CITATION.findall(groups)}, key=lambda p: int(p[1:]))
        valid   = [p for p in cited if p in self.passages]
        invalid = [p for p in cited if p not in self.passages]
        return {
            "answer": answer,
            "tool_calls": trace,
            "citations": {p: self.passages[p] for p in valid},
            "invalid_citations": invalid,
            "gating": self.gating,
            "model": model_used,
            "latency_ms": {"total": (time.perf_counter() - t_start) * 1000, "llm": llm_ms, "tools": tool_ms},
        }


def format_sources(citations):
    lines = []
    for pid, ch in citations.items():
        srcs = "; ".join(f"{s.get('publisher', '')} - {s.get('url', '')}" for s in ch["sources"])
        lines.append(f"  [{pid}] {ch['crop']} / {ch['disease']} / {ch['section']}: {srcs}")
    return "\n".join(lines)
