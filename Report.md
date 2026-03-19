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
| 1024 | | | | |
| 1536 | | | | |
| 2048 | | | | |
| 2560 | | | | |
| 3072 | | | | |

**Graph:** *(Insert GFlop/s vs matrix size graph here)*

**Analysis:**  
[TODO: Discuss observed performance, impact of B transposition in C++, comparison with Java.]

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
| 1024 | | | | |
| 1536 | | | | |
| 2048 | | | | |
| 2560 | | | | |
| 3072 | | | | |

**Results (large sizes — C++ only):**

| Size | C++ Time (s) | C++ GFlop/s |
|------|-------------|-------------|
| 4096 | | |
| 6144 | | |
| 8192 | | |
| 10240 | | |

**Graph:** *(Insert GFlop/s vs matrix size graph here — both languages for small sizes, C++ extended range)*

**Analysis:**  
[TODO: Compare performance gain versus 1.1, discuss L1/L2 cache exploitation, scaling behaviour for large matrices.]

---

### 1.3 — Block Matrix Multiplication

**Description:**  
Matrices are divided into sub-blocks of a configurable size. Computation follows the ikj order within each block, maximising data reuse within the cache before moving to the next block. Only implemented in C++ (`OnMultBlock` in `mult.cpp`).

**Tested matrix sizes:** 4096×4096 to 10240×10240 (step 2048)  
**Tested block sizes:** 128, 256, 512

**Results:**

| Size | Block=128 Time (s) | Block=128 GFlop/s | Block=256 Time (s) | Block=256 GFlop/s | Block=512 Time (s) | Block=512 GFlop/s |
|------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| 4096 | | | | | | |
| 6144 | | | | | | |
| 8192 | | | | | | |
| 10240 | | | | | | |

**Graph:** *(Insert GFlop/s vs matrix size for each block size)*

**Analysis:**  
[TODO: Identify optimal block size, explain relationship between block size and cache capacity, compare against 1.2.]

---

### 1.4 — Performance Counter Analysis with `perf`

**Description:**  
Linux `perf stat` was used to collect hardware performance counters for the three algorithm variants, focusing on L1 and L2 cache miss rates to validate the cache behaviour observed in execution times.

**Commands used:**
```bash
# General stats
perf stat ./a.out

# Cache miss counters
perf stat -e mem_load_retired.l1_miss,mem_load_retired.l2_miss ./a.out

# [TODO: Add any additional counters explored, e.g. cache-references, instructions, cycles]
```

**Results:**

| Algorithm | Matrix Size | L1 Misses | L2 Misses | [Other counter] |
|-----------|-------------|-----------|-----------|-----------------|
| ijk (1.1) | | | | |
| ikj (1.2) | | | | |
| Block (1.3) | | | | |

**Analysis:**  
[TODO: Correlate cache miss counts with execution times. Explain how ikj and block algorithms reduce cache misses compared to ijk.]

---

### 1.5 — Conclusions (Part 1)

[TODO: Summarise key findings from Part 1. Address:
- Performance difference between ijk and ikj and why it occurs.
- Effect of blocking on cache utilisation.
- Comparison between C++ and Java performance and reasons for any gap.
- How perf counters confirm the theoretical expectations.]

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