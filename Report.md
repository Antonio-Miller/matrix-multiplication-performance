# CPD Project 1 — Performance Evaluation of Matrix Multiplication

**Course:** L.EIC – CPD | FEUP  
**Group:** T07G11  
**Members:**
- António Tavares (202008423),
- Francisco Bandeira (202208526), 
- José Reis (202307304)  

---

## Table of Contents

1. [Introduction](#introduction)
2. [Part 1 — Single-Core Performance Evaluation](#part-1--single-core-performance-evaluation)
   - [1.1 — Basic Matrix Multiplication (ijk)](#11--basic-matrix-multiplication-ijk)
   - [1.2 — Line (Cache-Friendly) Matrix Multiplication (ikj)](#12--line-cache-friendly-matrix-multiplication-ikj)
   - [1.3 — Block Matrix Multiplication](#13--block-matrix-multiplication)
   - [1.4 — Performance Counter Analysis with `perf`](#14--performance-counter-analysis-with-perf)
   - [1.5 — Conclusions (Part 1)](#15--conclusions-part-1)
3. [Part 2 — Multi-Core Performance Evaluation](#part-2--multi-core-performance-evaluation)
   - [2.1 — Parallel Versions of Version 1 (ijk) and Version 2 (ikj)](#21--parallel-versions-of-version-1-ijk-and-version-2-ikj)
   - [2.2 — Version 2 Scalability with Thread Count (8192×8192)](#22--version-2-scalability-with-thread-count-81928192)
   - [2.3 — Exploring OpenMP Directives](#23--exploring-openmp-directives)
   - [2.4 — Conclusions (Part 2)](#24--conclusions-part-2)
4. [Overall Conclusions](#overall-conclusions)

---

## Introduction

This report analyses the performance impact of the memory hierarchy and multi-core parallelism on matrix multiplication. Two implementations are compared — C/C++ and Java — using progressively more cache-friendly and parallelised algorithms. Performance is measured in GFlop/s (based on the computational complexity of 2n³ floating-point operations for an n×n matrix product) and supplemented with hardware performance counters collected via Linux `perf`.

---

## Part 1 — Single-Core Performance Evaluation

### 1.1 — Basic Matrix Multiplication (ijk)

**Description:**  
The standard triple-nested loop multiplying one row of matrix A by each column of matrix B (ijk order). This access pattern causes poor cache utilisation for matrix B due to column-wise access.

**Implementation notes:**
- C++ (`OnMult` in `mult.cpp`): Uses a transposed copy of B (`phbt`) to improve cache locality despite the ijk loop structure.
- Java (`onMult` in `Mult.java`): Direct ijk loop on row-major arrays.

**Tested matrix sizes:** 1024×1024, 1536×1536, 2048×2048, 2560×2560, 3072×3072

**Results:**

| Size | C++ Time (s) | C++ GFlop/s | Java Time (s) | Java GFlop/s |
|------|-------------|-------------|--------------|--------------|
| 1024 | 0.405 | 5.298 | 2.191 | 0.980 |
| 1536 | 1.508 | 4.806 | 8.141 | 0.890 |
| 2048 | 3.791 | 4.532 | 20.016 | 0.858 |
| 2560 | 7.984 | 4.202 | 57.610 | 0.582 |
| 3072 | 13.751 | 4.217 | 83.510 | 0.694 |

**Graph:**
![GFlop/s vs Matrix Size](graphs/01_part1_1_ijk_gflops_cpp_vs_java.png)

**Analysis:**  
C++ achieves approximately 4.2–5.3 GFlop/s across the tested sizes, while Java delivers only 0.6–1.0 GFlop/s — a roughly 5× gap. The key reason is that the C++ implementation pre-transposes matrix B before the multiplication, converting the column-wise access pattern into sequential row access. This significantly reduces cache misses despite the ijk loop order. Java's implementation accesses B column by column in its original row-major layout, generating a cache miss on virtually every element of B touched. As matrix size grows and B no longer fits in cache, performance in Java degrades sharply (notably from 2048 to 2560). C++ performance also degrades with size but far more gradually because transposing B amortises the cache penalty across the computation.

---

### 1.2 — Line (Cache-Friendly) Matrix Multiplication (ikj)

**Description:**  
The loop order is reorganised to ikj, so that for each element `A[i][k]`, the entire row `k` of B is traversed sequentially. This dramatically improves spatial locality and cache hit rates.

**Implementation notes:**
- C++ (`OnMultLine` in `mult.cpp`): Caches `pha[iRow + k]` as `aik` and iterates sequentially across the j dimension.
- Java (`onMultLine` in `Mult.java`): Equivalent ikj loop structure on flat arrays.

**Tested matrix sizes (C++ and Java):** 1024×1024 to 3072×3072 (step 512)  
**Tested matrix sizes (C++ only):** 4096×4096 to 10240×10240 (step 2048)

**Results (small sizes — both languages):**

| Size | C++ Time (s) | C++ GFlop/s | Java Time (s) | Java GFlop/s |
|------|-------------|-------------|--------------|--------------|
| 1024 | 0.337 | 6.370 | 0.399 | 5.385 |
| 1536 | 1.308 | 5.539 | 1.429 | 5.071 |
| 2048 | 3.559 | 4.827 | 3.881 | 4.427 |
| 2560 | 7.794 | 4.305 | 8.226 | 4.079 |
| 3072 | 13.534 | 4.284 | 14.376 | 4.033 |

**Results (large sizes — C++ only):**

| Size | C++ Time (s) | C++ GFlop/s |
|------|-------------|-------------|
| 4096 | 33.783 | 4.068 |
| 6144 | 113.224 | 4.097 |
| 8192 | 265.079 | 4.148 |
| 10240 | 514.030 | 4.178 |

**Graph:**
![ikj Small Sizes](graphs/02_part1_2_ikj_small_gflops_cpp_vs_java.png)
![ikj Large Sizes C++](graphs/03_part1_2_ikj_large_cpp_gflops.png)

**Analysis:**  
The ikj reordering produces a dramatic improvement for Java — from ~0.7–1.0 GFlop/s (ijk) to ~4.0–5.4 GFlop/s — nearly closing the gap with C++. Both languages now exhibit similar performance because both access B row by row, making effective use of cache lines loaded from memory. At smaller sizes (1024–1536), C++ still holds a modest edge (~6.4 vs ~5.4 GFlop/s), likely due to JVM overhead and JIT warm-up; as matrix size grows the two converge tightly. For large C++ sizes (4096–10240), performance remains remarkably stable around 4.0–4.2 GFlop/s, indicating that memory bandwidth, not cache capacity, is the dominant bottleneck for these sizes — the working set far exceeds L2/L3 cache but sequential access patterns maximise prefetcher efficiency.

---

### 1.3 — Block Matrix Multiplication

**Description:**  
Matrices are divided into sub-blocks of a configurable size. Computation follows the ikj order within each block, maximising data reuse within the cache before moving to the next block. Only implemented in C++ (`OnMultBlock` in `mult.cpp`).

**Tested matrix sizes:** 4096×4096 to 10240×10240 (step 2048)  
**Tested block sizes:** 128, 256, 512

**Results:**

| Size | Block=128 Time (s) | Block=128 GFlop/s | Block=256 Time (s) | Block=256 GFlop/s | Block=512 Time (s) | Block=512 GFlop/s |
|------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| 4096 | 19.893 | 6.909 | 18.640 | 7.373 | 18.907 | 7.269 |
| 6144 | 67.167 | 6.906 | 62.250 | 7.451 | 63.414 | 7.315 |
| 8192 | 157.774 | 6.969 | 151.432 | 7.261 | 150.134 | 7.324 |
| 10240 | 309.924 | 6.929 | 302.836 | 7.091 | 294.440 | 7.293 |

**Graph:**
![Block Multiplication](graphs/04_part1_3_block_cpp_gflops.png)

**Analysis:**  
Block multiplication consistently outperforms the plain ikj version by approximately 70% at these sizes (e.g. ~7.3 GFlop/s vs ~4.1 GFlop/s for 4096×4096). By working on sub-matrices small enough to reside in L2/L3 cache, the algorithm drastically reduces the number of main-memory accesses. Block size 256 gives the best or near-best results across all tested matrix sizes, making it the optimal choice on this hardware. Block size 128 is slightly worse, likely because the smaller blocks reduce reuse granularity and increase loop overhead. Block size 512 performs comparably to 256 at larger matrices but marginally worse at 4096 — at that size, 512-element blocks approach or exceed the L2 cache capacity, eroding the advantage. Performance is highly consistent across matrix sizes (6.9–7.5 GFlop/s regardless of n), confirming that blocking successfully decouples computation throughput from the total matrix footprint.

---

### 1.4 — Performance Counter Analysis with `perf`

**Description:**  
Linux `perf stat` was used to collect hardware performance counters for the three algorithm variants, focusing on L1 and L2 cache miss rates to validate the cache behaviour observed in execution times.

**Commands used:**
```bash
# General stats
perf stat ./multCPP

# Cache miss counters
perf stat -e cpu_core/mem_load_retired.l1_miss/,cpu_core/mem_load_retired.l2_miss/ ./multCPP

# [TODO: Add any additional counters explored, e.g. cache-references, instructions, cycles]
```

**Results (C++, matrix size 4096×4096):**
 
| Algorithm | L1 Misses | L2 Misses | IPC (core) | Backend Bound % | Retiring % |
|-----------|-----------|-----------|------------|-----------------|------------|
| ijk (transposed B) | 572,976,555 | 68,565,584 | 2.31 | 54.4% | 30.6% |
| ikj (line) | 5,516,968,490 | 562,573,095 | 3.05 | 39.2% | 43.1% |
| Block (block=256) | 5,472,243,009 | 25,630,098 | 4.68 | 26.2% | 65.2% |
 
**Results (Java, matrix size 4096×4096):**
 
| Algorithm | L1 Misses | L2 Misses | IPC (core) | Backend Bound % | Retiring % |
|-----------|-----------|-----------|------------|-----------------|------------|
| ijk | 67,187,367,080 | 64,042,731,983 | 0.55 | 87.1% | 9.4% |
| ikj (line) | 1,081,737,067 | 44,098,460 | 2.62 | 34.6% | 52.9% |
| Block (block=256) | 923,802,616 | 10,219,453 | 3.78 | 4.3% | 74.7% |

**Analysis:**  
The cache counter data strongly confirms the theoretical expectations.
 
For C++, the ijk version (with transposed B) has surprisingly few L1 misses (~573M) compared to ikj (~5.5B). This seems counterintuitive at first but makes sense: the transposed B ensures that both the i-loop iteration over A's row and the j-loop iteration over B_transposed's row are sequential; the total memory traffic is reduced because the transposition is a one-time cost. However, IPC is lower (2.31) and backend-bound stalls are higher (54.4%), indicating that even with transposition the access pattern creates some memory pressure. The ikj version generates more L1 misses because matrix A is read once per k-step across the entire j dimension — a less regular pattern relative to the transposed B approach for this size — but IPC improves (3.05), suggesting better instruction-level parallelism. Critically, the block version keeps L2 misses at just ~25.6M (vs ~563M for ikj), demonstrating that blocking successfully keeps the hot working set within L2 cache. The result is the highest IPC (4.68) and the lowest backend-bound fraction (26.2%), with 65.2% of cycles spent in productive Retiring — explaining why block multiplication is ~70% faster.
 
For Java, the ijk version suffers catastrophic cache behaviour: ~67B L1 misses and ~64B L2 misses at 4096×4096, meaning nearly every L1 miss cascades all the way to main memory. This explains the 87.1% backend-bound rate and near-zero IPC (0.55). The ikj reordering reduces L1 misses by 62× and L2 misses by 1450×, confirming that sequential access to B's rows is the critical fix. Java's block version reduces L2 misses a further 4× vs ikj, and the backend-bound fraction drops to just 4.3% with 74.7% retiring — making Java block multiplication effectively compute-bound, an impressive result.

---

### 1.5 — Conclusions (Part 1)

The experiments in Part 1 clearly demonstrate that memory access patterns, not raw compute throughput, dominate performance for matrix multiplication at the sizes tested.
 
**ijk vs ikj:** The loop reordering from ijk to ikj has a transformative effect in Java (~5.5× speedup) and a modest but measurable effect in C++ (~5–20% depending on size). The reason is straightforward: ijk access on an unmodified B matrix strides through memory column by column, generating a cache miss on nearly every element; ikj accesses B row-sequentially, maximising spatial locality and prefetcher effectiveness. C++'s transposed-B optimisation partially compensates in the ijk version, which is why the gap is smaller there.
 
**Effect of blocking:** Tiling with block size 256 yields a consistent ~70% throughput gain over plain ikj in C++ for large matrices. By ensuring that three sub-matrices (a block of A, a block of B, and the corresponding accumulator in C) fit in L2/L3 cache simultaneously, the algorithm converts expensive DRAM accesses into fast cache hits. The perf counters confirm this: L2 misses drop by 22× from ikj to block multiplication.
 
**C++ vs Java:** For ijk, C++ outperforms Java by ~5× due to the B-transposition optimisation and the absence of JVM overhead. For ikj and block, the gap narrows to under 10% for large sizes, showing that when access patterns are optimised, Java's JIT-compiled code approaches native performance for compute-intensive loops.
 
**perf counter confirmation:** The TMA breakdown consistently shows that backend-bound stalls (caused by memory latency) decrease monotonically as the algorithm improves cache behaviour — from 87% (Java ijk) down to 4.3% (Java block). Retiring fractions increase in the same direction, confirming that better cache utilisation allows the processor to execute useful work in a larger fraction of cycles.

---

## Part 2 — Multi-Core Performance Evaluation

### 2.1 — Parallel Versions of Version 1 (ijk) and Version 2 (ikj)

**Description:**  
Two parallel strategies are analysed for Version 1 (ijk) and the correct parallel approach is applied to Version 2 (ikj). All experiments use 4 threads and matrix sizes from 1024×1024 to 3072×3072 (step 512).

#### V1-A — `#pragma omp parallel for` on outer loop (ijk)

Parallelises the outermost `i` loop. Each thread handles a disjoint set of rows of C, so there are no data races. This is the correct parallel strategy for the ijk formulation.

```cpp
#pragma omp parallel for
for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
        temp = 0;
        for (int k = 0; k < n; k++)
            temp += A(i,k) * B(k,j);
        C(i,j) = temp;
    }
```

#### V1-B — `#pragma omp parallel` + `#pragma omp for` on inner k loop (ijk)

Parallelises the innermost `k` loop. Introduces a race condition on `temp` and `C[i][j]` as multiple threads accumulate into the same element. Included for analysis purposes.

```cpp
#pragma omp parallel
for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
        temp = 0;
        #pragma omp for
        for (int k = 0; k < n; k++)
            temp += A(i,k) * B(k,j);
        C(i,j) = temp;
    }
```

#### V2 — `#pragma omp parallel for` on outer loop (ikj)

Parallelises the outermost `i` loop of the cache-friendly ikj formulation. Combines good spatial locality with correct parallelism.

```cpp
#pragma omp parallel for
for (int i = 0; i < n; i++)
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
            C(i,j) += A(i,k) * B(k,j);
```

**Results (4 threads, matrix sizes 1024–3072, step 512):**

| Size | V1-A Time (s) | V1-A GFlop/s | V1-B Time (s) | V1-B GFlop/s | V2 Time (s) | V2 GFlop/s |
|------|--------------|--------------|--------------|--------------|------------|------------|
| 1024 | | | | | | |
| 1536 | | | | | | |
| 2048 | | | | | | |
| 2560 | | | | | | |
| 3072 | | | | | | |

**Speedup and Efficiency (relative to sequential ikj, 4 threads):**

| Size | V1-A Speedup | V1-A Efficiency | V2 Speedup | V2 Efficiency |
|------|-------------|-----------------|-----------|---------------|
| 1024 | | | | |
| 1536 | | | | |
| 2048 | | | | |
| 2560 | | | | |
| 3072 | | | | |

**Graph:** *(Insert GFlop/s and Speedup vs matrix size for V1-A, V1-B, V2)*

**Analysis:**  
[TODO: Compare V1-A vs V1-B — explain the race condition in V1-B and its effect on correctness and performance. Compare V1-A vs V2 — discuss how cache-friendliness affects parallel performance. Comment on speedup and efficiency values.]

---

### 2.2 — Version 2 Scalability with Thread Count (8192×8192)

**Description:**  
Version 2 (ikj, `#pragma omp parallel for` on outer loop) is executed on an 8192×8192 matrix with a varying number of threads: 4, 8, 12, 16, 20, 24. The goal is to analyse scalability and identify any performance saturation or degradation.

**Results:**

| Threads | Time (s) | GFlop/s | Speedup (vs 1 thread) |
|---------|---------|---------|----------------------|
| 1 (baseline) | | | 1.00 |
| 4 | | | |
| 8 | | | |
| 12 | | | |
| 16 | | | |
| 20 | | | |
| 24 | | | |

**Graph:** *(Insert GFlop/s and Speedup vs number of threads)*

**Analysis:**  
[TODO: Identify the point of diminishing returns. Discuss memory bandwidth as a potential bottleneck. Compare observed speedup against the ideal linear speedup.]

---

### 2.3 — Exploring OpenMP Directives

**Description:**  
Two additional OpenMP directives are explored on Version 2 (ikj) and compared against the baseline `#pragma omp parallel for`.

#### `#pragma omp for simd`

Combines loop parallelism with SIMD vectorisation. The compiler is instructed to both distribute iterations across threads and apply vector instructions within each thread's chunk.

```cpp
#pragma omp parallel for simd
for (int i = 0; i < n; i++)
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
            C(i,j) += A(i,k) * B(k,j);
```

#### `#pragma omp parallel for collapse(2)`

Collapses the two outermost loops (i and k) into a single loop before distributing iterations, increasing the granularity of the parallel work and potentially improving load balance.

```cpp
#pragma omp parallel for collapse(2)
for (int i = 0; i < n; i++)
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
            C(i,j) += A(i,k) * B(k,j);
```

**Results (matrix size 8192×8192, thread counts: 4, 8, 12, 16, 20, 24):**

| Threads | Baseline GFlop/s | `for simd` GFlop/s | `collapse(2)` GFlop/s |
|---------|-----------------|--------------------|-----------------------|
| 4 | | | |
| 8 | | | |
| 12 | | | |
| 16 | | | |
| 20 | | | |
| 24 | | | |

**Graph:** *(Insert GFlop/s vs number of threads for all three directives)*

**Analysis:**  
[TODO: Determine whether `simd` provides additional gains beyond thread-level parallelism. Assess whether `collapse(2)` improves load balance or memory access patterns. Explain results in terms of hardware capabilities.]

---

### 2.4 — Conclusions (Part 2)

[TODO: Summarise key findings from Part 2. Address:
- Why V1-A outperforms V1-B (race condition, synchronisation overhead).
- Why V2 achieves better parallel performance than V1-A (cache locality preserved across threads).
- Scalability characteristics and the role of memory bandwidth limits.
- Effectiveness of `simd` and `collapse(2)` directives relative to the baseline.]

---

## Overall Conclusions

[TODO: Provide an integrative summary across both parts. Discuss the interplay between memory hierarchy effects and parallelism. Identify which combination of techniques (algorithm order + block size + threads + directives) delivers the best overall performance on the tested hardware. Reflect on what the perf counters reveal about the underlying bottlenecks.]

---

*Report compiled for CPD Project 1 — FEUP, March 2026.*