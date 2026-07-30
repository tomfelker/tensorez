"""Command-line interface.

    tensorez run <recipe.toml> [--cache-dir DIR] [--pretty]
    tensorez validate <recipe.toml> [--cache-dir DIR] [--pretty]

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
        p.add_argument("--cache-dir", default="cache",
                       help="stage cache directory (default: ./cache)")
        p.add_argument("--pretty", action="store_true",
                       help="human-readable progress instead of JSONL")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    emitter = EventEmitter(pretty=args.pretty)
    pipeline: Pipeline | None = None
    try:
        recipe = load_recipe(args.recipe)
        pipeline = Pipeline(recipe, Path(args.cache_dir), emitter)
        if args.command == "validate":
            result = pipeline.validate()
            if args.pretty:
                print(f"recipe ok: {recipe.name} ({result['frame_count']} frames)")
                for s in result["stages"]:
                    state = "cache hit" if s["cached"] else "cache miss"
                    print(f"  {s['stage']}: {state}")
            else:
                emitter.emit("validate_result", **result)
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
