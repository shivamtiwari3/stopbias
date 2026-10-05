"""Command line: prep -> plan -> run -> report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def cmd_prep(a: argparse.Namespace) -> None:
    from .corpus import prepare

    metas = prepare(DATA / "corpus", n_pairs=a.pairs)
    users = sum(1 for m in metas if m.role == "user")
    print(f"prepared {users} user clips and {len(metas) - users} agent clips -> {DATA / 'corpus'}")


def cmd_plan(a: argparse.Namespace) -> None:
    from .trials import plan, save_plan

    trials = plan(DATA / "corpus" / "manifest.json", n_trials=a.pairs)
    save_plan(trials, DATA / "trials" / "plan.json")
    lat = [t.true_stop_latency_s * 1000 for t in trials]
    print(f"planned {len(trials)} trials; true stop latency {min(lat):.0f}-{max(lat):.0f} ms "
          f"-> {DATA / 'trials' / 'plan.json'}")


def cmd_run(a: argparse.Namespace) -> None:
    from .experiment import run

    rows = run(DATA / "trials" / "plan.json", RESULTS / "raw" / "phase1.jsonl",
               detectors=tuple(a.detectors), only=tuple(a.only) if a.only else None)
    print(f"wrote {len(rows)} measurements -> {RESULTS / 'raw' / 'phase1.jsonl'}")


def cmd_report(a: argparse.Namespace) -> None:
    from .conditions import REFERENCE
    from .stats import load, markdown_table, summarise

    df = load(RESULTS / "raw" / "phase1.jsonl")
    s = summarise(df)
    RESULTS.mkdir(parents=True, exist_ok=True)
    s.to_csv(RESULTS / "phase1_summary.csv", index=False)

    parts = ["# Phase 1 results\n",
             f"{len(df)} measurements over {df.trial_id.nunique()} trials, "
             f"{df.condition.nunique()} conditions, {df.detector.nunique()} detectors.\n",
             f"Reference condition: `{REFERENCE}`.\n"]
    for det in sorted(df.detector.unique()):
        for vp in sorted(df[df.detector == det].viewpoint.unique()):
            parts.append(f"\n## detector = `{det}`, viewpoint = `{vp}`\n\n")
            parts.append(markdown_table(s, det, vp))
    (RESULTS / "phase1_tables.md").write_text("".join(parts))
    print(json.dumps({"summary_csv": str(RESULTS / "phase1_summary.csv"),
                      "tables_md": str(RESULTS / "phase1_tables.md"),
                      "cells": len(s)}, indent=2))


def cmd_power(a: argparse.Namespace) -> None:
    from .power import analyse, markdown

    pw = analyse(RESULTS / "raw" / "phase1.jsonl")
    RESULTS.mkdir(parents=True, exist_ok=True)
    pw.to_csv(RESULTS / "phase2_power.csv", index=False)
    (RESULTS / "phase2_power.md").write_text(markdown(pw))
    print(f"wrote {RESULTS / 'phase2_power.md'} ({len(pw)} cells)")


def cmd_bridge(a: argparse.Namespace) -> None:
    from .bridge import BridgeConfig
    from .bridgecall import markdown, validate

    man = json.loads((DATA / "corpus" / "manifest.json").read_text())
    agent = next(c["path"] for c in man["clips"] if c["role"] == "agent")
    user = next(c["path"] for c in man["clips"] if c["role"] == "user")
    cfg = BridgeConfig(model_sr=a.model_sr, jitter_frames=a.jitter_frames)

    df = validate(agent, user, cfg=cfg)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "phase2_bridge.csv", index=False)
    (RESULTS / "phase2_bridge.md").write_text(markdown(df))

    ok = df[df.err_ms.notna()]
    model = ok[ok.viewpoint == "model"].err_ms
    print(f"bridge delay: {cfg.delay_ms:.0f} ms "
          f"({cfg.inbound_delay_ms:.0f} inbound + {cfg.outbound_delay_ms:.0f} outbound)")
    print(f"model-tap error over {len(model)} recoveries: median {model.median():+.1f} ms, "
          f"range {model.min():+.0f} to {model.max():+.0f}")
    print(f"wrote {RESULTS / 'phase2_bridge.md'} ({len(df)} recoveries)")


def cmd_plots(a: argparse.Namespace) -> None:
    from .plots import make_all

    paths = make_all(RESULTS / "raw" / "phase1.jsonl", RESULTS / "figures")
    print("\n".join(str(p) for p in paths))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stopbias")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep", help="download and cut the corpus")
    p.add_argument("--pairs", type=int, default=60)
    p.set_defaults(fn=cmd_prep)

    p = sub.add_parser("plan", help="freeze the trial plan")
    p.add_argument("--pairs", type=int, default=60)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("run", help="measure every cell")
    p.add_argument("--detectors", nargs="+", default=["silero", "energy"])
    p.add_argument("--only", nargs="*", default=None, help="restrict to these condition names")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="aggregate to CSV + markdown")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("power", help="simulate Phase 2 sample sizes from Phase 1 residuals")
    p.set_defaults(fn=cmd_power)

    p = sub.add_parser("bridge", help="validate the SIP<->WebSocket bridge offline")
    p.add_argument("--model-sr", type=int, default=24000,
                   help="model socket rate: 24000 for GPT-Realtime, 16000 for Gemini Live")
    p.add_argument("--jitter-frames", type=int, default=2, help="fixed playout depth, in frames")
    p.set_defaults(fn=cmd_bridge)

    p = sub.add_parser("plots", help="render figures")
    p.set_defaults(fn=cmd_plots)

    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
