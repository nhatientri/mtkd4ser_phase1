#!/usr/bin/env python3
"""
summarize_results.py – one table comparing every finished run (accuracy + efficiency).

Reads SAVE_DIR/results/*.json written at the end of each run by pipeline.Trainer.test().

    python summarize_results.py --SAVE_DIR checkpoints            # markdown table
    python summarize_results.py --SAVE_DIR checkpoints --csv out.csv
    python summarize_results.py --SAVE_DIR checkpoints --group     # mean ± std over seeds/folds
"""

import argparse
import glob
import json
import os
import re

import pandas as pd


def _student(mc: dict) -> str:
    return str(mc.get("backbone", "")).rstrip("/").split("/")[-1] + (f"-L{mc['num_layers']}" if mc.get("num_layers") else "")


def load(save_dir: str) -> pd.DataFrame:
    rows = []
    for fp in sorted(glob.glob(os.path.join(save_dir, "results", "*.json"))):
        with open(fp) as f:
            r = json.load(f)
        c, t, e = r.get("config", {}), r.get("test", {}), r.get("efficiency") or {}
        rows.append({
            "run": r["run_name"],
            "paradigm": c.get("PARADIGM"),
            "routing": c.get("ROUTING") if c.get("PARADIGM") == "MTKD" else "-",
            "data": c.get("LINGUALITY"),
            "student": _student(r.get("model_config") or {"backbone": c.get("BACKBONE", ""),
                                                          "num_layers": c.get("NUM_LAYERS")}),
            "seed": c.get("SEED"),
            "fold": f"S{c.get('SESSION')}F{c.get('CAFE_FOLD')}",
            "params_M": r.get("params_total", 0) / 1e6,
            "test_UR": t.get("UR"),
            "test_UR_en": t.get("UR_en"),
            "test_UR_fr": t.get("UR_fr"),
            "test_UR_lang_mean": t.get("UR_lang_mean"),
            "test_WR": t.get("WR"),
            "best_epoch": r.get("best_epoch"),
            "sec_per_epoch": e.get("avg_epoch_train_sec"),
            "train_samples_per_s": e.get("train_samples_per_sec"),
            "total_train_min": (e.get("total_train_sec") or 0) / 60,
            "teacher_cache_min": (e.get("teacher_cache_sec") or 0) / 60,
            "peak_mem_GB": (e.get("peak_gpu_mem_mb") or 0) / 1024 or None,
            "test_ms_per_clip": r.get("test_ms_per_clip"),
            "device": r.get("device"),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--SAVE_DIR", default="/content/checkpoints" if os.path.isdir("/content") else "./checkpoints")
    p.add_argument("--csv", default=None)
    p.add_argument("--group", action="store_true", help="aggregate over seeds and folds")
    p.add_argument("--filter", default=None, help="regex on run name")
    a = p.parse_args()

    df = load(a.SAVE_DIR)
    if df.empty:
        print(f"No results in {a.SAVE_DIR}/results/")
        return
    if a.filter:
        df = df[df["run"].str.contains(re.compile(a.filter))]

    if a.group:
        keys = ["paradigm", "routing", "data", "student"]
        num = [c for c in df.columns if df[c].dtype.kind in "fi" and c not in ("seed",)]
        g = df.groupby(keys)[num]
        mean, std, n = g.mean(), g.std(), g.size().rename("n_runs")
        out = mean.copy()
        for c in ("test_UR", "test_UR_en", "test_UR_fr", "test_UR_lang_mean"):
            if c in out:
                out[c] = [f"{m:.4f} ± {s:.4f}" if pd.notna(s) else f"{m:.4f}" for m, s in zip(mean[c], std[c])]
        df = out.join(n).reset_index()

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    try:
        print(df.to_markdown(index=False, floatfmt=".4f"))
    except ImportError:  # `tabulate` not installed
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if a.csv:
        df.to_csv(a.csv, index=False)
        print(f"\nSaved {a.csv}")


if __name__ == "__main__":
    main()
