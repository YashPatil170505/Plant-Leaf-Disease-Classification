"""
metrics.py - Local scorers for the RAG evaluation (no LLM judge)

  NLI cross-encoder        -> is each answer sentence entailed by a retrieved passage?
                              (citation support + hallucination rate)
  Relevance cross-encoder  -> does the answer address the question?

Both are small models that run on the local GPU/CPU. NLI on paraphrased text is
conservative: a correct paraphrase is sometimes scored "neutral", so the
hallucination rate here is an upper-bound estimate. Spot-check flagged sentences.
"""

import re
import numpy as np
from sentence_transformers import CrossEncoder

NLI_MODEL       = "cross-encoder/nli-deberta-v3-base"
RELEVANCE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ENTAIL_THRESHOLD = 0.5

# Sentences that are guardrail / meta statements, not claims from the references
META = re.compile(
    r"(label|extension|consult|diagnostic lab|reference library|doesn't cover|does not cover|"
    r"not a substitute|clearer photo|plain background|lab photos|field photos|classifier|"
    r"the model|confidence|uncertain|not sure|possible conditions|report)", re.I)
ABSTAIN = re.compile(
    r"(doesn't|does not|don't|do not) (cover|have|contain|include)|not covered|no information|"
    r"isn't (in|covered)|not in my|outside (of )?my", re.I)
CITE = re.compile(r"\s*\[P\d+(?:\s*,\s*P\d+)*\]")


def split_sentences(text):
    text = re.sub(r"^\s*[-*\d.]+\s+", "", text, flags=re.M)        # list markers
    text = text.replace("**", "")
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.split()) >= 5]


class Scorer:
    def __init__(self):
        self.nli = CrossEncoder(NLI_MODEL)
        labels = {v.lower(): int(k) for k, v in self.nli.model.config.id2label.items()}
        self.entail_idx = labels["entailment"]
        self.rel = CrossEncoder(RELEVANCE_MODEL)

    def entailment(self, premises, hypothesis):
        """Max entailment probability of hypothesis over the given premises."""
        if not premises:
            return 0.0
        # Small NLI models degrade on long premises, so score against 1-2 sentence
        # windows of each passage and take the max (SummaC-style).
        windows = []
        for p in premises:
            sents = [s for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]
            windows += sents + [" ".join(sents[i:i + 2]) for i in range(len(sents) - 1)]
        logits = self.nli.predict([(w, hypothesis) for w in windows])
        logits = np.atleast_2d(logits)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return float(probs[:, self.entail_idx].max())

    def relevance(self, question, answer):
        """Sigmoid of the MS-MARCO relevance logit, in [0, 1]."""
        return float(1 / (1 + np.exp(-self.rel.predict([(question, CITE.sub("", answer))])[0])))

    def score_answer(self, question, answer, retrieved, cited):
        """retrieved / cited: dicts of passage id -> passage text."""
        sentences = []
        for s in split_sentences(answer):
            ids = re.findall(r"P\d+", " ".join(re.findall(r"\[(P\d+(?:\s*,\s*P\d+)*)\]", s)))
            clean = CITE.sub("", s).strip()
            if META.search(clean) and not ids:
                continue
            all_p = self.entailment(list(retrieved.values()), clean)
            cit_p = self.entailment([cited[i] for i in ids if i in cited], clean) if ids else None
            sentences.append({"sentence": clean, "cites": ids,
                              "supported_by_any": all_p >= ENTAIL_THRESHOLD, "p_any": round(all_p, 3),
                              "supported_by_cited": None if cit_p is None else cit_p >= ENTAIL_THRESHOLD,
                              "p_cited": None if cit_p is None else round(cit_p, 3)})
        return {
            "relevance": self.relevance(question, answer),
            "abstained": bool(ABSTAIN.search(answer)),
            "sentences": sentences,
            "n_claims": len(sentences),
            "n_unsupported": sum(not s["supported_by_any"] for s in sentences),
            "n_cited_claims": sum(bool(s["cites"]) for s in sentences),
            "n_cited_supported": sum(bool(s["supported_by_cited"]) for s in sentences),
        }
