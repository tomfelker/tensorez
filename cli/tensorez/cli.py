"""Command-line interface.

    tensorez run <recipe.toml> [--runs-dir DIR] [--cache-dir DIR] [--events]
    tensorez validate <recipe.toml> [--runs-dir DIR] [--cache-dir DIR] [--events]

Recipe paths resolve against the working directory, and so do these two
options, which default to ``tensorez_runs/`` and ``tensorez_cache/`` right
there.  Both hold bulk data that is regenerable and often large, so they are
options rather than recipe keys: point them at a fast scratch disk once
(the GUI has a setting for it) and every recipe's archives and cache go
there, leaving the capture drive holding only recipes and results.

Output is human-readable by default; ``--events`` switches stdout to the
JSONL event stream (CONTRACT.md §2), which is what the GUI asks for.

Exit code 0 on success; nonzero after emitting an ``error`` event.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from .events import EventEmitter
from .pipeline import Pipeline, PipelineError
from .recipe import RecipeError, load_recipe


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensorez", description="TensoRez Next — planetary lucky-imaging stacker"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("run", "run the full pipeline"),
        ("validate", "parse + resolve the recipe and report per-stage cache hit/miss; no work"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("recipe", help="path to the recipe .toml")
        p.add_argument("--runs-dir", default="tensorez_runs",
                       help="archive of past runs; each lands in "
                            "<runs-dir>/<recipe name>/<timestamp>/ "
                            "(default: ./tensorez_runs)")
        p.add_argument("--cache-dir", default="tensorez_cache",
                       help="stage cache directory (default: ./tensorez_cache)")
        p.add_argument("--events", action="store_true",
                       help="emit the JSONL event stream instead of human-readable progress")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # Windows still defaults stdout to the ANSI code page, which mangles (or
    # raises on) the arcsec marks and dashes in our messages once output is
    # piped.  The JSONL side escapes non-ASCII, but humans get it raw.
    for stream in (sys.stdout, sys.stderr):
        if getattr(stream, "reconfigure", None) is not None:
            stream.reconfigure(encoding="utf-8", errors="replace")
    emitter = EventEmitter(pretty=not args.events)
    pipeline: Pipeline | None = None
    try:
        recipe = load_recipe(args.recipe)
        cache_dir = Path(args.cache_dir).resolve()
        runs_dir = Path(args.runs_dir).resolve()
        pipeline = Pipeline(recipe, cache_dir, runs_dir, emitter)
        if args.command == "validate":
            result = pipeline.validate()
            if args.events:
                emitter.emit("validate_result", **result)
            else:
                print(f"recipe ok: {recipe.name} ({result['frame_count']} frames)")
                print(f"  working: {Path.cwd()}")
                print(f"  output:  {recipe.output_dir}")
                print(f"  runs:    {runs_dir / recipe.name}")
                print(f"  cache:   {cache_dir}")
                for s in result["stages"]:
                    state = "cache hit" if s["cached"] else "cache miss"
                    print(f"  {s['stage']}: {state}")
            return 0
        pipeline.run()
        return 0
    except RecipeError as e:
        emitter.error(f"invalid recipe: {e}")
        return 1
    except PipelineError as e:
        emitter.error(str(e), stage=e.stage)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # unexpected: include traceback for the bug report
        stage = pipeline.current_stage if pipeline is not None else None
        emitter.error(f"{type(e).__name__}: {e}", stage=stage, traceback=traceback.format_exc())
        return 1
    finally:
        emitter.close()


if __name__ == "__main__":
    sys.exit(main())
