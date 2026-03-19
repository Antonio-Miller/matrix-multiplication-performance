import java.util.Scanner;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

public class Mult {

    static double computeGflops(int n, double elapsedSeconds) {
        return 2.0 * n * n * n / (elapsedSeconds * 1e9);
    }

    static void printResultPreview(double[] phc, int mBr) {
        System.out.print("Result preview (first row, up to 10 values): ");
        for (int j = 0; j < Math.min(10, mBr); j++) {
            System.out.print(phc[j] + " ");
        }
        System.out.println();
    }

    static void printRunSummary(String algorithm, int n, double elapsedSeconds,
                                int threads, int blockSize, boolean raceWarning) {
        double gflops = computeGflops(n, elapsedSeconds);

        System.out.println("\n=== Run Summary ===");
        System.out.println("Algorithm  : " + algorithm);
        System.out.println("Matrix     : " + n + "x" + n);
        if (threads > 0) {
            System.out.println("Threads    : " + threads);
        }
        if (blockSize > 0) {
            System.out.println("Block size : " + blockSize);
        }
        System.out.printf("Time (s)   : %.6f%n", elapsedSeconds);
        System.out.printf("GFlop/s    : %.3f%n", gflops);
        System.out.printf("CSV        : %s,%d,%s,%s,%.6f,%.3f%n",
                algorithm,
                n,
                threads > 0 ? Integer.toString(threads) : "-",
                blockSize > 0 ? Integer.toString(blockSize) : "-",
                elapsedSeconds,
                gflops);
        if (raceWarning) {
            System.out.println("Warning    : This variant has a race condition; values may be incorrect.");
        }
        System.out.println("===================\n");
    }

    static void onMult(int mAr, int mBr) {
        double[] pha = new double[mAr * mAr];
        double[] phb = new double[mAr * mAr];
        double[] phc = new double[mAr * mAr];

        for (int i = 0; i < mAr; i++)
            for (int j = 0; j < mAr; j++)
                pha[i * mAr + j] = 1.0;

        for (int i = 0; i < mBr; i++)
            for (int j = 0; j < mBr; j++)
                phb[i * mBr + j] = (double)(i + 1);

        long time1 = System.nanoTime();

        for (int i = 0; i < mAr; i++) {
            for (int j = 0; j < mBr; j++) {
                double temp = 0;
                for (int k = 0; k < mAr; k++) {
                    temp += pha[i * mAr + k] * phb[k * mBr + j];
                }
                phc[i * mAr + j] = temp;
            }
        }

        long time2 = System.nanoTime();
        double elapsed = (time2 - time1) / 1e9;
        printRunSummary("P1-V1-ijk", mAr, elapsed, -1, -1, false);
        printResultPreview(phc, mBr);
    }

    // Line-by-line (cache-friendly ikj) matrix multiplication
    static void onMultLine(int mAr, int mBr) {
        double[] pha = new double[mAr * mAr];
        double[] phb = new double[mAr * mAr];
        double[] phc = new double[mAr * mAr];

        for (int i = 0; i < mAr; i++)
            for (int j = 0; j < mAr; j++)
                pha[i * mAr + j] = 1.0;

        for (int i = 0; i < mBr; i++)
            for (int j = 0; j < mBr; j++)
                phb[i * mBr + j] = (double)(i + 1);

        for (int i = 0; i < mAr; i++)
            for (int j = 0; j < mBr; j++)
                phc[i * mAr + j] = 0.0;

        long time1 = System.nanoTime();

        /*
         * for i  → pick a row of A
         *   for k  → pick one element A[i][k]
         *     for j  → multiply across entire row k of B
         *               C[i][j] += A[i][k] * B[k][j]
         */
        for (int i = 0; i < mAr; i++) {
            for (int k = 0; k < mAr; k++) {
                for (int j = 0; j < mBr; j++) {
                    phc[i * mAr + j] += pha[i * mAr + k] * phb[k * mBr + j];
                }
            }
        }

        long time2 = System.nanoTime();
        double elapsed = (time2 - time1) / 1e9;
        printRunSummary("P1-V2-ikj(line)", mAr, elapsed, -1, -1, false);
        printResultPreview(phc, mBr);
    }

    // Block matrix multiplication
    static void onMultBlock(int mAr, int mBr, int bkSize) {
        double[] pha = new double[mAr * mAr];
        double[] phb = new double[mAr * mAr];
        double[] phc = new double[mAr * mAr];

        for (int i = 0; i < mAr; i++)
            for (int j = 0; j < mAr; j++)
                pha[i * mAr + j] = 1.0;

        for (int i = 0; i < mBr; i++)
            for (int j = 0; j < mBr; j++)
                phb[i * mBr + j] = (double)(i + 1);

        // Initialize C to zero
        for (int i = 0; i < mAr; i++)
            for (int j = 0; j < mBr; j++)
                phc[i * mAr + j] = 0.0;

        long time1 = System.nanoTime();

        for (int ii = 0; ii < mAr; ii += bkSize) {
            for (int kk = 0; kk < mAr; kk += bkSize) {
                for (int jj = 0; jj < mBr; jj += bkSize) {
                    for (int i = ii; i < Math.min(ii + bkSize, mAr); i++) {
                        for (int k = kk; k < Math.min(kk + bkSize, mAr); k++) {
                            for (int j = jj; j < Math.min(jj + bkSize, mBr); j++) {
                                phc[i * mAr + j] += pha[i * mAr + k] * phb[k * mBr + j];
                            }
                        }
                    }
                }
            }
        }

        long time2 = System.nanoTime();
        double elapsed = (time2 - time1) / 1e9;
        printRunSummary("P1-V3-block", mAr, elapsed, -1, bkSize, false);
        printResultPreview(phc, mBr);
    }

    // Part 2.1 – Version 1 (ijk), Variant A: parallel outer i loop (#pragma omp parallel for)
    static void onMultParallelV1A(int mAr, int mBr, int numThreads) throws Exception {
        double[] pha = new double[mAr * mAr];
        double[] phb = new double[mAr * mAr];
        double[] phc = new double[mAr * mAr];

        for (int i = 0; i < mAr; i++)
            for (int j = 0; j < mAr; j++)
                pha[i * mAr + j] = 1.0;
        for (int i = 0; i < mBr; i++)
            for (int j = 0; j < mBr; j++)
                phb[i * mBr + j] = (double)(i + 1);

        long t1 = System.nanoTime();
        try (ForkJoinPool pool = new ForkJoinPool(numThreads)) {
            pool.submit(() ->
                IntStream.range(0, mAr).parallel().forEach(i -> {
                    for (int j = 0; j < mBr; j++) {
                        double temp = 0;
                        for (int k = 0; k < mAr; k++)
                            temp += pha[i * mAr + k] * phb[k * mBr + j];
                        phc[i * mAr + j] = temp;
                    }
                })
            ).get();
        }

        long t2 = System.nanoTime();
        double elapsed = (t2 - t1) / 1e9;
        printRunSummary("P2-V1A-ijk-parallel-for", mAr, elapsed, numThreads, -1, false);
        printResultPreview(phc, mBr);
    }

    // Part 2.1 – Version 1 (ijk), Variant B: parallel inner k loop
    static void onMultParallelV1B(int mAr, int mBr, int numThreads) throws Exception {
        double[] pha = new double[mAr * mAr];
        double[] phb = new double[mAr * mAr];
        double[] phc = new double[mAr * mAr];

        for (int i = 0; i < mAr; i++)
            for (int j = 0; j < mAr; j++)
                pha[i * mAr + j] = 1.0;
        for (int i = 0; i < mBr; i++)
            for (int j = 0; j < mBr; j++)
                phb[i * mBr + j] = (double)(i + 1);

        long t1 = System.nanoTime();

        // All threads run all (i,j) iterations; k range split per thread -> race on phc[i*mAr+j]
        try (ForkJoinPool pool = new ForkJoinPool(numThreads)) {
            pool.submit(() ->
                IntStream.range(0, numThreads).parallel().forEach(threadId -> {
                    int kChunk = (mAr + numThreads - 1) / numThreads;
                    int kStart = threadId * kChunk;
                    int kEnd = Math.min(kStart + kChunk, mAr);
                    for (int i = 0; i < mAr; i++) {
                        for (int j = 0; j < mBr; j++) {
                            double temp = 0;
                            for (int k = kStart; k < kEnd; k++)
                                temp += pha[i * mAr + k] * phb[k * mBr + j];
                            phc[i * mAr + j] += temp; // race condition (same as C++ version)
                        }
                    }
                })
            ).get();
        }

        long t2 = System.nanoTime();
        double elapsed = (t2 - t1) / 1e9;
        printRunSummary("P2-V1B-ijk-inner-k", mAr, elapsed, numThreads, -1, true);
        printResultPreview(phc, mBr);
    }

    // Part 2.1 – Version 2 (ikj), parallel outer i loop
    static void onMultParallelV2(int mAr, int mBr, int numThreads) throws Exception {
        double[] pha = new double[mAr * mAr];
        double[] phb = new double[mAr * mAr];
        double[] phc = new double[mAr * mAr];

        for (int i = 0; i < mAr; i++)
            for (int j = 0; j < mAr; j++)
                pha[i * mAr + j] = 1.0;
        for (int i = 0; i < mBr; i++)
            for (int j = 0; j < mBr; j++)
                phb[i * mBr + j] = (double)(i + 1);

        long t1 = System.nanoTime();
        try (ForkJoinPool pool = new ForkJoinPool(numThreads)) {
            pool.submit(() ->
                IntStream.range(0, mAr).parallel().forEach(i -> {
                    for (int k = 0; k < mAr; k++)
                        for (int j = 0; j < mBr; j++)
                            phc[i * mAr + j] += pha[i * mAr + k] * phb[k * mBr + j];
                })
            ).get();
        }

        long t2 = System.nanoTime();
        double elapsed = (t2 - t1) / 1e9;
        printRunSummary("P2-V2-ikj-parallel-for", mAr, elapsed, numThreads, -1, false);
        printResultPreview(phc, mBr);
    }

    public static void main(String[] args) throws Exception {
        Scanner sc = new Scanner(System.in);
        int op;

        do {
            System.out.println();
            System.out.println("── Part 1 ──────────────────────────");
            System.out.println("1. Multiplication");
            System.out.println("2. Line Multiplication");
            System.out.println("3. Block Multiplication");
            System.out.println("── Part 2 ──────────────────────────");
            System.out.println("4. Parallel V1-A  (ijk, parallel for outer)");
            System.out.println("5. Parallel V1-B  (ijk, parallel inner k)");
            System.out.println("6. Parallel V2    (ikj, parallel for outer)");
            System.out.println("────────────────────────────────────");
            System.out.println("0. Exit");
            System.out.print("Selection?: ");
            op = sc.nextInt();

            if (op == 0) break;

            System.out.print("Dimensions: lins=cols ? ");
            int lin = sc.nextInt();
            int col = lin;

            switch (op) {
                case 1:
                    onMult(lin, col);
                    break;
                case 2:
                    onMultLine(lin, col);
                    break;
                case 3:
                    System.out.print("Block Size? ");
                    int blockSize = sc.nextInt();
                    onMultBlock(lin, col, blockSize);
                    break;
                case 4:
                    System.out.print("Number of threads? ");
                    int t4 = sc.nextInt();
                    onMultParallelV1A(lin, col, t4);
                    break;
                case 5:
                    System.out.print("Number of threads? ");
                    int t5 = sc.nextInt();
                    onMultParallelV1B(lin, col, t5);
                    break;
                case 6:
                    System.out.print("Number of threads? ");
                    int t6 = sc.nextInt();
                    onMultParallelV2(lin, col, t6);
                    break;
            }
        } while (op != 0);

        sc.close();
    }
}
