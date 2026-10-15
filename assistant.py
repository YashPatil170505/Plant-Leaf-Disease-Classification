"""
assistant.py - Chat with the plant disease assistant (LLM + RAG + tool calling)

Usage:
    python assistant.py --image path/to/leaf.jpg      # diagnose, then ask follow-ups
    python assistant.py                               # ask general questions
    python assistant.py --image leaf.jpg --ask "How do I prevent it?"   # one-shot

In the chat, type 'report' to generate a diagnosis report, 'quit' to exit.
Requires: GOOGLE_API_KEY in .env, and the index built with  python -m rag.build_index
"""

import sys
import argparse
from rag.agent import PlantAssistant, format_sources


def show(result):
    print(f"\nAssistant: {result['answer']}\n")
    if result["citations"]:
        print("Sources:")
        print(format_sources(result["citations"]))
    if result["invalid_citations"]:
        print(f"  WARNING: answer cited unknown passages {result['invalid_citations']}")
    lat = result["latency_ms"]
    print(f"\n  ({result['model']} | total {lat['total']:.0f} ms, LLM {lat['llm']:.0f} ms, "
          f"local tools {lat['tools']:.0f} ms)\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")      # Windows console: degree signs, dashes
    parser = argparse.ArgumentParser(description="Plant disease assistant")
    parser.add_argument("--image", help="Leaf image to diagnose first")
    parser.add_argument("--ask",   help="Ask a single question and exit")
    args = parser.parse_args()

    bot = PlantAssistant()
    print("Plant disease assistant. Type 'report' for a diagnosis report, 'quit' to exit.\n")

    first = None
    if args.image:
        first = f"I uploaded a leaf image: {args.image}. What's wrong with this plant?"
        if args.ask:
            first += f" Also: {args.ask}"
    elif args.ask:
        first = args.ask

    if first:
        print(f"You: {first}")
        show(bot.ask(first))
        if args.ask:
            raise SystemExit

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in {"quit", "exit"}:
            break
        if q.lower() == "report":
            q = "Please generate a diagnosis report for this leaf."
        show(bot.ask(q))
