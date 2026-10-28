"""Local model runner and model-free cached-score decoding."""
import argparse
import json
from pathlib import Path
import sys
from .decoder import decode
from .snapshot import Snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description="One-pass d1-2B named-entity list extraction")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Download the pinned official q8 model")
    download.add_argument("--model-dir")
    verify = sub.add_parser("verify-model", help="Validate pinned host code and optionally the full weights")
    verify.add_argument("--model-dir")
    verify.add_argument("--full-weights", action="store_true")
    for command in ["extract", "capture", "decode"]:
        p = sub.add_parser(command)
        if command == "decode":
            p.add_argument("--scores", required=True)
        else:
            source = p.add_mutually_exclusive_group(required=True)
            source.add_argument("--text")
            source.add_argument("--input", help="UTF-8 text file, or - for stdin")
            p.add_argument("--model-dir")
            p.add_argument("--threads", type=int, default=8)
            p.add_argument("--max-input-tokens", type=int, default=8192)
        if command == "capture":
            p.add_argument("--scores", required=True)
        else:
            p.add_argument("--kinds", nargs="+", choices=["people", "products", "organizations", "combined"])
            p.add_argument("--policy", choices=["atomic", "soft", "baseline"], default="atomic")
            p.add_argument("--cutoff", type=float, default=.003)
            p.add_argument("--max-items", type=int, default=10)
            p.add_argument("--trace", action="store_true")
        p.add_argument("--output", help="JSON output file; default stdout")
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            from .model_files import download_model
            print(download_model(args.model_dir))
            return 0
        if args.command == "verify-model":
            from .model_files import default_model_dir, validate_model
            print(validate_model(args.model_dir or default_model_dir(), full_weights=args.full_weights))
            return 0
        if args.command == "decode":
            snapshot = Snapshot.load(args.scores)
        else:
            from .backend import ListExtractor
            text = args.text if args.text is not None else (sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8"))
            with ListExtractor(args.model_dir, threads=args.threads, max_input_tokens=args.max_input_tokens) as model:
                snapshot = model.capture(text)
        if args.command == "capture":
            snapshot.save(args.scores)
            result = {"saved_scores": args.scores, "usage": snapshot.metadata}
        else:
            result = decode(snapshot, kinds=args.kinds, policy=args.policy, cutoff=args.cutoff, max_items=args.max_items, trace=args.trace)
            if args.command == "extract":
                result["usage"] = snapshot.metadata
        value = json.dumps(result, indent=2, ensure_ascii=False)
        if args.output:
            Path(args.output).write_text(value + "\n", encoding="utf-8")
        else:
            print(value)
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(2, f"d1-list: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
