"""
run_cross_analysis.py — Main CLI entry point for cross-pipeline analysis.

Usage:
    python run_cross_analysis.py [options]

Options:
    --nlp-output PATH       Override NLP pipeline output directory
    --network-output PATH   Override network pipeline output directory
    --output-dir PATH       Where to write results (default: ./output)
    --insights LIST         Comma-separated: i1,i2,i3,i4,i5,i6  (default: all)
    --year-start INT        Filter data from this year
    --year-end INT          Filter data up to this year
    --no-charts             Skip PNG chart generation (CSV outputs only)
    --log-level LEVEL       DEBUG / INFO / WARNING (default: INFO)

Example:
    python run_cross_analysis.py --insights i1,i3 --year-start 2005
    python run_cross_analysis.py --nlp-output ../arms-control-nlp/output
"""
from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import time
from pathlib import Path

# ── Make the cross-analysis package importable regardless of cwd ──────────────
sys.path.insert(0, str(Path(__file__).parent))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-reference NLP pipeline results with network results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
          Insights:
            i1  Hypocrisy Fingerprint      — which NLP dims predict ethical risk?
            i2  Community–Rhetoric         — do trade communities cluster in NLP space?
            i3  ATT Language vs Compliance — does ATT language predict fewer violations?
            i4  Moral Foundations          — does care/harm rhetoric anti-correlate with violations?
            i5  Transition Lag             — rhetoric or transfers change first after regime shift?
            i6  P5 Hypocrisy Profile       — P5 vs. other major exporters on NLP dimensions
        """),
    )
    p.add_argument("--nlp-output",     type=Path, default=None)
    p.add_argument("--network-output", type=Path, default=None)
    p.add_argument("--output-dir",     type=Path, default=None)
    p.add_argument("--insights",       type=str,  default="all",
                   help="Comma-separated insight IDs (i1..i6) or 'all'")
    p.add_argument("--year-start",     type=int,  default=None)
    p.add_argument("--year-end",       type=int,  default=None)
    p.add_argument("--no-charts",      action="store_true")
    p.add_argument("--log-level",      type=str,  default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_separator(char: str = "-", width: int = 70) -> None:
    print(char * width)


def _print_join_report(report: dict) -> None:
    _print_separator()
    print("  JOIN REPORT")
    _print_separator()
    print(f"  Total country-year rows : {report['total_country_years']}")
    print(f"  In both pipelines       : {report['in_both_pipelines']}")
    print(f"  NLP only                : {report['nlp_only']}")
    print(f"  Unique countries        : {report['unique_countries']}")
    if report.get("year_range"):
        print(f"  Year range              : {report['year_range'][0]}-{report['year_range'][1]}")
    print(f"  NLP columns added       : {report['nlp_cols_added']}")
    print(f"  Network columns added   : {report['net_cols_added']}")
    print(f"  Total master columns    : {report['total_columns']}")
    _print_separator()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)

    # ── Build config ──────────────────────────────────────────────────────────
    from config import Config
    overrides = {}
    if args.nlp_output:
        overrides["nlp_output_dir"] = args.nlp_output
    if args.network_output:
        overrides["net_output_dir"] = args.network_output
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.year_start:
        overrides["year_start"] = args.year_start
    if args.year_end:
        overrides["year_end"] = args.year_end
    if args.no_charts:
        overrides["produce_charts"] = False

    cfg = Config(**overrides)
    cfg.charts_dir.mkdir(parents=True, exist_ok=True)
    cfg.csvs_dir.mkdir(parents=True, exist_ok=True)

    print()
    _print_separator("=")
    print("  CROSS-PIPELINE ANALYSIS")
    _print_separator("=")
    print(f"  NLP output dir     : {cfg.nlp_output_dir}")
    print(f"  Network output dir : {cfg.net_output_dir}")
    print(f"  Results dir        : {cfg.output_dir}")
    print(f"  Charts             : {'disabled' if args.no_charts else 'enabled'}")
    if args.year_start or args.year_end:
        print(f"  Year filter        : {args.year_start or '—'} → {args.year_end or '—'}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────────
    from loader import load_everything
    import pandas as pd

    t0 = time.time()
    print("Loading data...")
    master, report, nlp_dfs, net_dfs, bridge = load_everything(cfg)
    print(f"  Done in {time.time() - t0:.1f}s")
    _print_join_report(report)

    # Save master dataset
    master_path = cfg.csvs_dir / "master_merged.csv"
    master.to_csv(master_path, index=False)
    print(f"  Master dataset saved: {master_path}")
    print()

    # ── Determine which insights to run ──────────────────────────────────────
    from analyses.i1_hypocrisy_fingerprint    import HypocrisyFingerprint
    from analyses.i2_community_rhetoric        import CommunityRhetoric
    from analyses.i3_att_language_compliance   import AttLanguageCompliance
    from analyses.i4_moral_foundations_violations import MoralFoundationsViolations
    from analyses.i5_transition_lag            import RegimeTransitionLag
    from analyses.i6_p5_profile                import P5HypocrisyProfile

    all_analyses = {
        "i1": HypocrisyFingerprint(cfg),
        "i2": CommunityRhetoric(cfg),
        "i3": AttLanguageCompliance(cfg),
        "i4": MoralFoundationsViolations(cfg),
        "i5": RegimeTransitionLag(cfg),
        "i6": P5HypocrisyProfile(cfg),
    }

    if args.insights.lower() == "all":
        selected = list(all_analyses.keys())
    else:
        selected = [s.strip().lower() for s in args.insights.split(",")]
        unknown = [s for s in selected if s not in all_analyses]
        if unknown:
            print(f"  WARNING: unknown insight IDs: {unknown}")
            selected = [s for s in selected if s in all_analyses]

    # ── Run analyses ──────────────────────────────────────────────────────────
    results: dict[str, dict] = {}
    for aid in selected:
        analysis = all_analyses[aid]
        t1 = time.time()
        result = analysis.run_safe(master, nlp_dfs, net_dfs)
        elapsed = time.time() - t1
        results[aid] = result
        status = result.get("status", "?")
        icon = "OK" if status == "ok" else "FAIL"
        print(f"  {icon} {aid} ({analysis.name}) [{elapsed:.1f}s]")
        if result.get("error"):
            print(f"    Error: {result['error'][:120]}...")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    _print_separator("=")
    print("  SUMMARY")
    _print_separator("=")
    ok_count = sum(1 for r in results.values() if r.get("status") == "ok")
    print(f"  Completed: {ok_count}/{len(results)} analyses\n")

    for aid, result in results.items():
        name = all_analyses[aid].name
        print(f"  [{aid}] {name}")
        for finding in result.get("key_findings", []):
            # Encode to ASCII, replacing unsupported chars, to avoid Windows cp1252 errors
            safe = finding.encode("ascii", errors="replace").decode("ascii")
            wrapped = textwrap.fill(safe, width=65, initial_indent="    - ",
                                    subsequent_indent="      ")
            print(wrapped)
        print()

    _print_separator("=")
    print(f"  Outputs written to: {cfg.output_dir}")
    _print_separator("=")
    print()


if __name__ == "__main__":
    main()
