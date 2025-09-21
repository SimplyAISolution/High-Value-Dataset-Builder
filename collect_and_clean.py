#!/usr/bin/env python3
"""
High-Value Dataset Builder - Collection and Cleaning Pipeline

Provides a modular, well-documented CLI for:
  1) Parsing CLI arguments for --config/--input/--output/--log-usage
  2) Loading YAML config with source, cleaning, and export settings
  3) Collecting files via glob patterns
  4) Cleaning CSV/JSONL/JSON (dedupe, trim whitespace, normalize unicode, remove empties)
  5) Exporting to CSV/JSONL/Parquet
  6) Optional usage logging

Designed for clarity and composability to maximize community value.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # Loaded lazily with helpful error if used

# -----------------------------
# Data classes for configuration
# -----------------------------
@dataclass
class SourceConfig:
    path: str
    include: List[str] = field(default_factory=lambda: ["*.jsonl", "*.json", "*.csv"])

@dataclass
class CleaningConfig:
    drop_duplicates: bool = True
    trim_whitespace: bool = True
    normalize_unicode: bool = True
    remove_empty: bool = True
    lower_keys: bool = False

@dataclass
class ExportConfig:
    format: str = "parquet"  # csv|jsonl|parquet
    output_dir: str = "./data/processed"
    filename: str = "dataset"

@dataclass
class AppConfig:
    sources: List[SourceConfig] = field(default_factory=lambda: [SourceConfig(path="./data/raw")])
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

# -----------------------------
# Utility functions
# -----------------------------

def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# YAML loading
# -----------------------------

def load_config(config_path: Optional[Path]) -> AppConfig:
    """Load configuration from YAML if provided, else defaults.

    The YAML may define keys: sources, cleaning, export. Unknown keys are ignored.
    """
    if config_path is None:
        return AppConfig()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if yaml is None:
        raise RuntimeError("pyyaml is required to load YAML configs. Install with `pip install pyyaml`." )

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Parse sources
    sources_raw = data.get("sources") or data.get("source")
    sources: List[SourceConfig] = []
    if isinstance(sources_raw, list):
        for s in sources_raw:
            if isinstance(s, dict):
                sources.append(SourceConfig(path=str(s.get("path", "./data/raw")), include=list(s.get("include", ["*.jsonl", "*.json", "*.csv"]))))
    elif isinstance(sources_raw, dict):
        sources.append(SourceConfig(path=str(sources_raw.get("path", "./data/raw")), include=list(sources_raw.get("include", ["*.jsonl", "*.json", "*.csv"]))))
    else:
        sources.append(SourceConfig(path="./data/raw"))

    # Parse cleaning
    cl = data.get("cleaning", {}) or {}
    cleaning = CleaningConfig(
        drop_duplicates=bool(cl.get("drop_duplicates", True)),
        trim_whitespace=bool(cl.get("trim_whitespace", True)),
        normalize_unicode=bool(cl.get("normalize_unicode", True)),
        remove_empty=bool(cl.get("remove_empty", True)),
        lower_keys=bool(cl.get("lower_keys", False)),
    )

    # Parse export
    ex = data.get("export", {}) or {}
    export = ExportConfig(
        format=str(ex.get("format", "parquet")).lower(),
        output_dir=str(ex.get("output_dir", "./data/processed")),
        filename=str(ex.get("filename", "dataset")),
    )

    return AppConfig(sources=sources, cleaning=cleaning, export=export)


# -----------------------------
# File discovery (glob collect)
# -----------------------------

def collect_files(sources: Sequence[SourceConfig]) -> List[Path]:
    """Collect files from each source via glob includes."""
    hits: List[Path] = []
    for src in sources:
        base = Path(src.path)
        for pattern in (src.include or ["*"]):
            for p in base.rglob(pattern):
                if p.is_file():
                    hits.append(p)
    # Stable order
    hits = sorted(set(hits))
    return hits


# -----------------------------
# Readers for CSV/JSON/JSONL
# -----------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
                else:
                    rows.append({"value": obj})
            except json.JSONDecodeError:
                eprint(f"Skipping invalid JSON line in {path}")
    return rows


def read_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        # ensure dicts
        rows: List[Dict[str, Any]] = []
        for x in data:
            if isinstance(x, dict):
                rows.append(x)
            else:
                rows.append({"value": x})
        return rows
    elif isinstance(data, dict):
        return [data]
    else:
        return [{"value": data}]


def read_csv(path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def infer_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".jsonl":
        return "jsonl"
    if ext == ".json":
        return "json"
    if ext == ".csv":
        return "csv"
    return "unknown"


def load_records(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for p in paths:
        fmt = infer_format(p)
        if fmt == "jsonl":
            records.extend(read_jsonl(p))
        elif fmt == "json":
            records.extend(read_json(p))
        elif fmt == "csv":
            records.extend(read_csv(p))
        else:
            eprint(f"Skipping unsupported file type: {p}")
    return records


# -----------------------------
# Cleaning operations
# -----------------------------

def normalize_text(val: Any) -> Any:
    if isinstance(val, str):
        # NFC normalization keeps composed form; good default for text
        return unicodedata.normalize("NFC", val)
    return val


def trim_ws(val: Any) -> Any:
    if isinstance(val, str):
        return val.strip()
    return val


def remove_empty_record(rec: Dict[str, Any]) -> bool:
    # Keep if any non-empty scalar value
    for v in rec.values():
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        if isinstance(v, (list, dict)) and len(v) == 0:
            continue
        return True
    return False


def lower_keys(rec: Dict[str, Any]) -> Dict[str, Any]:
    return { (k.lower() if isinstance(k, str) else k): v for k, v in rec.items() }


def clean_records(records: List[Dict[str, Any]], cfg: CleaningConfig) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen_hashes: set = set()

    for rec in records:
        r = dict(rec)
        if cfg.lower_keys:
            r = lower_keys(r)

        # Field-wise transforms
        if cfg.normalize_unicode:
            r = {k: normalize_text(v) for k, v in r.items()}
        if cfg.trim_whitespace:
            r = {k: trim_ws(v) for k, v in r.items()}

        if cfg.remove_empty and not remove_empty_record(r):
            continue

        if cfg.drop_duplicates:
            # Hash a stable JSON representation for dedupe
            try:
                key = json.dumps(r, sort_keys=True, ensure_ascii=False)
            except TypeError:
                # Fallback: convert non-serializable to str
                key = json.dumps({k: (str(v) if not isinstance(v, (str, int, float, bool, type(None), list, dict)) else v)
                                   for k, v in r.items()}, sort_keys=True, ensure_ascii=False)
            if key in seen_hashes:
                continue
            seen_hashes.add(key)

        out.append(r)

    return out


# -----------------------------
# Writers for CSV/JSONL/Parquet
# -----------------------------

def write_jsonl(records: Sequence[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_json(records: Sequence[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(records), f, ensure_ascii=False, indent=2)


def write_csv(records: Sequence[Dict[str, Any]], path: Path) -> None:
    if not records:
        # Create empty file
        path.write_text("", encoding="utf-8")
        return
    df = pd.DataFrame.from_records(records)
    df.to_csv(path, index=False)


def write_parquet(records: Sequence[Dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame.from_records(records)
    # Parquet engine: pyarrow or fastparquet
    df.to_parquet(path, index=False)


def export_records(records: List[Dict[str, Any]], cfg: ExportConfig) -> Path:
    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)

    fmt = cfg.format.lower()
    if fmt not in {"csv", "jsonl", "parquet", "json"}:
        eprint(f"Unknown export format '{cfg.format}', defaulting to parquet")
        fmt = "parquet"

    out_path = out_dir / f"{cfg.filename}.{ 'json' if fmt=='json' else fmt }"
    if fmt == "jsonl":
        write_jsonl(records, out_path)
    elif fmt == "csv":
        write_csv(records, out_path)
    elif fmt == "json":
        write_json(records, out_path)
    else:  # parquet
        write_parquet(records, out_path)

    return out_path


# -----------------------------
# Usage logging
# -----------------------------

def log_usage(enabled: bool, info: Dict[str, Any], path: Path = Path("usage.log")) -> None:
    if not enabled:
        return
    record = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect, clean, and export datasets.")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    p.add_argument("--input", type=str, default=None, help="Override input directory (replaces sources in config)")
    p.add_argument("--output", type=str, default=None, help="Override output directory")
    p.add_argument("--format", type=str, default=None, help="Override export format: csv|jsonl|parquet|json")
    p.add_argument("--filename", type=str, default=None, help="Override output filename (without extension)")
    p.add_argument("--log-usage", action="store_true", help="Append run metadata to usage.log")
    return p


def apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    # Input override
    if args.input:
        cfg.sources = [SourceConfig(path=args.input)]
    # Output override
    if args.output:
        cfg.export.output_dir = args.output
    # Format override
    if args.format:
        cfg.export.format = args.format
    # Filename override
    if args.filename:
        cfg.export.filename = args.filename
    return cfg


def run_pipeline(cfg: AppConfig, log_enabled: bool) -> Tuple[int, int, Path]:
    files = collect_files(cfg.sources)
    n_files = len(files)

    records_in = load_records(files)
    n_in = len(records_in)

    cleaned = clean_records(records_in, cfg.cleaning)
    n_out = len(cleaned)

    out_path = export_records(cleaned, cfg.export)

    log_usage(log_enabled, {
        "files": n_files,
        "n_records_in": n_in,
        "n_records_out": n_out,
        "config": {
            "sources": [s.__dict__ for s in cfg.sources],
            "cleaning": cfg.cleaning.__dict__,
            "export": cfg.export.__dict__,
        },
        "output": str(out_path),
    })

    return n_in, n_out, out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config) if args.config else None
    cfg = load_config(config_path)
    cfg = apply_overrides(cfg, args)

    try:
        n_in, n_out, out_path = run_pipeline(cfg, args.log_usage)
    except Exception as e:
        eprint(f"Pipeline failed: {e}")
        return 1

    print(f"Processed {n_in} records -> {n_out} cleaned records")
    print(f"Output written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
