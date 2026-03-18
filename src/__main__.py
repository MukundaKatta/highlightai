"""CLI for highlightai."""
import sys, json, argparse
from .core import Highlightai

def main():
    parser = argparse.ArgumentParser(description="HighlightAI — Sports Highlight Generator. Auto-generate highlight reels from full game footage.")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Highlightai()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.generate(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"highlightai v0.1.0 — HighlightAI — Sports Highlight Generator. Auto-generate highlight reels from full game footage.")

if __name__ == "__main__":
    main()
