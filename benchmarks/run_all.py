"""Run every benchmark against both baseline/ and fork/ and write a report.

Each benchmark runs in a fresh subprocess with PYTHONPATH pointed at one
variant, so the two copies of the package can never interfere.

Usage:  python benchmarks/run_all.py [--tag LABEL]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

BENCHES = [
    "bench_read.py",
    "bench_stats.py",
    "bench_validate.py",
    "bench_gzip.py",
    "bench_memmap.py",
    "bench_write.py",
]
VARIANTS = ("baseline", "fork")


def run_one(bench: str, variant: str) -> dict | None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(ROOT, variant), HERE])
    env["MRC_VARIANT"] = variant
    # Runs one of this directory's own benchmark scripts, never external input
    proc = subprocess.run(  # noqa: S603
        [sys.executable, os.path.join(HERE, bench)],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if proc.returncode != 0:
        print(f"  !! {bench} [{variant}] failed:\n{proc.stderr[-800:]}")
        return None
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


def gmt_stamp() -> str:
    """Timestamp in GMT (UTC+0)."""
    now = datetime.now(timezone.utc).astimezone(timezone(timedelta(0)))
    return now.strftime("%Y-%m-%d %H:%M:%S GMT")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run", help="label for this run")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    collected: dict[str, dict[str, dict]] = {}

    for bench in BENCHES:
        print(f"running {bench}")
        for variant in VARIANTS:
            out = run_one(bench, variant)
            if out is None:
                continue
            collected.setdefault(out["bench"], {})[variant] = out["results"]

    payload = {
        "tag": args.tag,
        "timestamp_gmt": gmt_stamp(),
        "platform": {
            "python": sys.version.split()[0],
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "cpu_count": os.cpu_count(),
        },
        "benchmarks": collected,
    }

    json_path = os.path.join(RESULTS, f"{args.tag}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    md_path = os.path.join(RESULTS, f"{args.tag}.md")
    with open(md_path, "w") as f:
        f.write(render(payload))

    print(f"\nwrote {json_path}\nwrote {md_path}\n")
    print(render(payload))


def render(payload: dict) -> str:
    p = payload["platform"]
    lines = [
        "# Benchmark results",
        "",
        f"**Run:** `{payload['tag']}`  ",
        f"**Timestamp:** {payload['timestamp_gmt']}  ",
        f"**Host:** Python {p['python']}, {p['machine']}, {p['cpu_count']} core(s)",
        "",
        "Best of N wall-clock; `peak` is transient allocation above the resident",
        "array, measured by `tracemalloc` on a separate untimed call.",
        "",
        "| Operation | Baseline | Fork | Speedup | Peak base | Peak fork |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for bench in sorted(payload["benchmarks"]):
        variants = payload["benchmarks"][bench]
        base = variants.get("baseline", {})
        fork = variants.get("fork", {})
        for op in base:
            b = base[op]
            k = fork.get(op)
            if k is None:
                lines.append(
                    f"| `{op}` | {b['ms']:.1f} ms | - | - | "
                    f"{b['peak_mib']:.1f} MiB | - |"
                )
                continue
            speed = b["ms"] / k["ms"] if k["ms"] else float("nan")
            flag = " **" if speed >= 1.15 else ""
            lines.append(
                f"| `{op}` | {b['ms']:.1f} ms | {k['ms']:.1f} ms |"
                f"{flag}{speed:.2f}x{flag.strip()} | "
                f"{b['peak_mib']:.1f} MiB | {k['peak_mib']:.1f} MiB |"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
