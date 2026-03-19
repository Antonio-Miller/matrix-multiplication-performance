#!/usr/bin/env python3
"""
PSEUDOCODE PLAN (detailed):
1) Ensure file starts with a valid Python shebang as the very first line.
2) Remove any shell command text accidentally pasted into the Python source.
   - Bad text found: "python3 generate_report_graphs.py"
   - Keep that only as documentation in the docstring usage examples, not executable code.
3) Keep the existing graph-generation pipeline:
   a) Parse CLI arguments for optional C++ CSV, Java CSV, and output directory.
   b) If CSV paths are not provided, resolve latest matching files in ../results.
   c) Load CSV rows into a typed Row dataclass structure.
   d) Build plots required by report sections:
      - Part 1.1 ijk comparison (C++ vs Java)
      - Part 1.2 ikj small sizes comparison
      - Part 1.2 ikj large sizes (C++ only)
      - Part 1.3 block-size comparison (C++)
      - Part 2.1 parallel variants at 4 threads (C++ and Java)
      - Part 2.1 speedup/efficiency vs sequential baseline
      - Part 2.2 scaling for V2 at n=8192 (GFlop/s and speedup)
   e) Save generated images in timestamped output folder.
   f) Write an index text file listing generated graph files.
4) Preserve original behavior and structure; only fix syntax-breaking line corruption.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt


@dataclass
class Row:
    algorithm: str
    size: int
    threads: Optional[int]
    block_size: Optional[int]
    time_s: float
    gflops: float


def parse_optional_int(value: str) -> Optional[int]:
    value = value.strip()
    if not value or value == "-":
        return None
    return int(value)


def load_rows(csv_path: Path) -> List[Row]:
    rows: List[Row] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(
                Row(
                    algorithm=raw["algorithm"].strip(),
                    size=int(raw["size"]),
                    threads=parse_optional_int(raw["threads"]),
                    block_size=parse_optional_int(raw["block_size"]),
                    time_s=float(raw["time_s"]),
                    gflops=float(raw["gflops"]),
                )
            )
    return rows


def latest_csv(results_dir: Path, prefix: str) -> Path:
    matches = sorted(results_dir.glob(f"{prefix}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No files found matching {prefix}_*.csv in {results_dir}")
    return matches[-1]


def filter_rows(
    rows: Sequence[Row],
    algo_prefix: str,
    threads: Optional[int] = None,
    block_size: Optional[int] = None,
) -> List[Row]:
    out: List[Row] = []
    for r in rows:
        if not r.algorithm.startswith(algo_prefix):
            continue
        if threads is not None and r.threads != threads:
            continue
        if block_size is not None and r.block_size != block_size:
            continue
        out.append(r)
    out.sort(key=lambda x: x.size)
    return out


def to_xy(rows: Sequence[Row], y_attr: str) -> tuple[List[int], List[float]]:
    xs = [r.size for r in rows]
    ys = [float(getattr(r, y_attr)) for r in rows]
    return xs, ys


def save_plot(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_time_lookup(rows: Sequence[Row], algo_prefix: str, threads: Optional[int] = None) -> Dict[int, float]:
    d: Dict[int, float] = {}
    for r in filter_rows(rows, algo_prefix, threads=threads):
        d[r.size] = r.time_s
    return d


def plot_if_data(fig: plt.Figure, out_path: Path, generated: List[Path], has_data: bool, reason: str) -> None:
    if has_data:
        save_plot(fig, out_path)
        generated.append(out_path)
    else:
        plt.close(fig)
        print(f"[skip] {out_path.name}: {reason}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    results_dir = root_dir / "results"

    parser = argparse.ArgumentParser(description="Generate report graphs from benchmark CSV files.")
    parser.add_argument("--cpp-csv", type=Path, help="Path to C++ CSV file. Defaults to latest cpp_report_*.csv")
    parser.add_argument("--java-csv", type=Path, help="Path to Java CSV file. Defaults to latest java_report_*.csv")
    parser.add_argument("--output-dir", type=Path, help="Output directory for graphs")
    args = parser.parse_args()

    cpp_csv = args.cpp_csv or latest_csv(results_dir, "cpp_report")
    java_csv = args.java_csv or latest_csv(results_dir, "java_report")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (results_dir / f"graphs_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using C++ CSV : {cpp_csv}")
    print(f"Using Java CSV: {java_csv}")
    print(f"Output folder : {output_dir}")

    cpp = load_rows(cpp_csv)
    java = load_rows(java_csv)

    generated: List[Path] = []

    # 1.1 Basic ijk - C++ vs Java
    c = filter_rows(cpp, "P1-V1-ijk")
    j = filter_rows(java, "P1-V1-ijk")
    fig, ax = plt.subplots(figsize=(8, 5))
    if c:
        x, y = to_xy(c, "gflops")
        ax.plot(x, y, marker="o", label="C++ ijk")
    if j:
        x, y = to_xy(j, "gflops")
        ax.plot(x, y, marker="s", label="Java ijk")
    ax.set_title("Part 1.1 - Basic ijk (GFlop/s)")
    ax.set_xlabel("Matrix size (n)")
    ax.set_ylabel("GFlop/s")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_if_data(fig, output_dir / "01_part1_1_ijk_gflops_cpp_vs_java.png", generated, bool(c or j), "no ijk data")

    # 1.2 ikj small sizes - C++ vs Java
    c = [r for r in filter_rows(cpp, "P1-V2-ikj(line)") if r.size <= 3072]
    j = [r for r in filter_rows(java, "P1-V2-ikj(line)") if r.size <= 3072]
    fig, ax = plt.subplots(figsize=(8, 5))
    if c:
        x, y = to_xy(c, "gflops")
        ax.plot(x, y, marker="o", label="C++ ikj (small)")
    if j:
        x, y = to_xy(j, "gflops")
        ax.plot(x, y, marker="s", label="Java ikj (small)")
    ax.set_title("Part 1.2 - ikj Small Sizes (GFlop/s)")
    ax.set_xlabel("Matrix size (n)")
    ax.set_ylabel("GFlop/s")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_if_data(fig, output_dir / "02_part1_2_ikj_small_gflops_cpp_vs_java.png", generated, bool(c or j), "no small ikj data")

    # 1.2 ikj large sizes - C++ only
    c = [r for r in filter_rows(cpp, "P1-V2-ikj(line)") if r.size >= 4096]
    fig, ax = plt.subplots(figsize=(8, 5))
    if c:
        x, y = to_xy(c, "gflops")
        ax.plot(x, y, marker="o", label="C++ ikj (large)")
    ax.set_title("Part 1.2 - ikj Large Sizes C++ (GFlop/s)")
    ax.set_xlabel("Matrix size (n)")
    ax.set_ylabel("GFlop/s")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_if_data(fig, output_dir / "03_part1_2_ikj_large_cpp_gflops.png", generated, bool(c), "no large C++ ikj data")

    # 1.3 block multiplication - C++
    fig, ax = plt.subplots(figsize=(8, 5))
    block_any = False
    for b, marker in [(128, "o"), (256, "s"), (512, "^")]:
        r = filter_rows(cpp, "P1-V3-block", block_size=b)
        if not r:
            continue
        x, y = to_xy(r, "gflops")
        ax.plot(x, y, marker=marker, label=f"Block={b}")
        block_any = True
    ax.set_title("Part 1.3 - Block Multiplication C++ (GFlop/s)")
    ax.set_xlabel("Matrix size (n)")
    ax.set_ylabel("GFlop/s")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_if_data(fig, output_dir / "04_part1_3_block_cpp_gflops.png", generated, block_any, "no block data")

    # 2.1 - GFlop/s vs size for C++ and Java (4 threads)
    for rows, name in [(cpp, "cpp"), (java, "java")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        a = filter_rows(rows, "P2-V1A-ijk-parallel-for", threads=4)
        b = filter_rows(rows, "P2-V1B-ijk-inner-k", threads=4)
        v2 = filter_rows(rows, "P2-V2-ikj-parallel-for", threads=4)
        has_any = bool(a or b or v2)
        if a:
            x, y = to_xy(a, "gflops")
            ax.plot(x, y, marker="o", label="V1-A")
        if b:
            x, y = to_xy(b, "gflops")
            ax.plot(x, y, marker="s", label="V1-B")
        if v2:
            x, y = to_xy(v2, "gflops")
            ax.plot(x, y, marker="^", label="V2")
        ax.set_title(f"Part 2.1 - {name.upper()} 4 Threads (GFlop/s)")
        ax.set_xlabel("Matrix size (n)")
        ax.set_ylabel("GFlop/s")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plot_if_data(
            fig,
            output_dir / f"05_part2_1_{name}_gflops_vs_size.png",
            generated,
            has_any,
            f"no part 2.1 data ({name})",
        )

    # 2.1 - Speedup and efficiency for V1-A and V2 (relative to sequential ikj)
    for rows, name in [(cpp, "cpp"), (java, "java")]:
        seq = build_time_lookup(rows, "P1-V2-ikj(line)")
        t_v1a = build_time_lookup(rows, "P2-V1A-ijk-parallel-for", threads=4)
        t_v2 = build_time_lookup(rows, "P2-V2-ikj-parallel-for", threads=4)

        common_sizes = sorted(set(seq).intersection(t_v1a).intersection(t_v2))
        if not common_sizes:
            print(f"[skip] 06_part2_1_{name}_speedup_efficiency.png: missing common sizes")
            continue

        speed_v1a = [seq[s] / t_v1a[s] for s in common_sizes]
        speed_v2 = [seq[s] / t_v2[s] for s in common_sizes]
        eff_v1a = [v / 4.0 for v in speed_v1a]
        eff_v2 = [v / 4.0 for v in speed_v2]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
        ax1.plot(common_sizes, speed_v1a, marker="o", label="V1-A speedup")
        ax1.plot(common_sizes, speed_v2, marker="^", label="V2 speedup")
        ax1.set_title(f"Part 2.1 - {name.upper()} Speedup")
        ax1.set_xlabel("Matrix size (n)")
        ax1.set_ylabel("Speedup")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(common_sizes, eff_v1a, marker="o", label="V1-A efficiency")
        ax2.plot(common_sizes, eff_v2, marker="^", label="V2 efficiency")
        ax2.set_title(f"Part 2.1 - {name.upper()} Efficiency (4 threads)")
        ax2.set_xlabel("Matrix size (n)")
        ax2.set_ylabel("Efficiency")
        ax2.set_ylim(bottom=0)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        save_plot(fig, output_dir / f"06_part2_1_{name}_speedup_efficiency.png")
        generated.append(output_dir / f"06_part2_1_{name}_speedup_efficiency.png")

    # 2.2 - V2 scaling with thread count at n=8192 (both languages)
    def v2_scaling(rows: Sequence[Row]) -> List[Row]:
        out = [r for r in rows if r.algorithm.startswith("P2-V2-ikj-parallel-for") and r.size == 8192 and r.threads is not None]
        out.sort(key=lambda r: r.threads or math.inf)
        return out

    cpp_v2 = v2_scaling(cpp)
    java_v2 = v2_scaling(java)

    fig, ax = plt.subplots(figsize=(8, 5))
    if cpp_v2:
        ax.plot([r.threads for r in cpp_v2], [r.gflops for r in cpp_v2], marker="o", label="C++")
    if java_v2:
        ax.plot([r.threads for r in java_v2], [r.gflops for r in java_v2], marker="s", label="Java")
    ax.set_title("Part 2.2 - V2 Scaling (8192x8192) GFlop/s")
    ax.set_xlabel("Threads")
    ax.set_ylabel("GFlop/s")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_if_data(fig, output_dir / "07_part2_2_v2_scaling_gflops.png", generated, bool(cpp_v2 or java_v2), "no V2 scaling data")

    fig, ax = plt.subplots(figsize=(8, 5))
    has_speed = False
    if cpp_v2:
        t1 = next((r.time_s for r in cpp_v2 if r.threads == 1), None)
        if t1:
            threads = [r.threads for r in cpp_v2]
            speed = [t1 / r.time_s for r in cpp_v2]
            ax.plot(threads, speed, marker="o", label="C++ speedup")
            has_speed = True
    if java_v2:
        t1 = next((r.time_s for r in java_v2 if r.threads == 1), None)
        if t1:
            threads = [r.threads for r in java_v2]
            speed = [t1 / r.time_s for r in java_v2]
            ax.plot(threads, speed, marker="s", label="Java speedup")
            has_speed = True

    if cpp_v2 or java_v2:
        all_threads = sorted({r.threads for r in cpp_v2 + java_v2 if r.threads is not None})
        if all_threads:
            ax.plot(all_threads, all_threads, linestyle="--", label="Ideal linear")

    ax.set_title("Part 2.2 - V2 Scaling (8192x8192) Speedup")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup vs 1 thread")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_if_data(fig, output_dir / "08_part2_2_v2_scaling_speedup.png", generated, has_speed, "missing 1-thread baseline")

    # Write index file for quick reference.
    index_path = output_dir / "graphs_index.txt"
    with index_path.open("w", encoding="utf-8") as f:
        f.write(f"C++ CSV: {cpp_csv}\n")
        f.write(f"Java CSV: {java_csv}\n")
        f.write("Generated graphs:\n")
        for p in generated:
            f.write(f"- {p.name}\n")
    generated.append(index_path)

    print("\nGenerated files:")
    for p in generated:
        print(f"- {p}")


if __name__ == "__main__":
    main()