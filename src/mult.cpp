#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <omp.h>

using namespace std;

#define SYSTEMTIME clock_t

double computeGflops(int n, double elapsedSeconds)
{
    return 2.0 * n * n * n / (elapsedSeconds * 1e9);
}

void printResultPreview(const double* phc, int m_br)
{
    cout << "Result preview (first row, up to 10 values): ";
    for (int j = 0; j < min(10, m_br); j++)
        cout << phc[j] << " ";
    cout << endl;
}

void printRunSummary(const string& algorithm, int n, double elapsedSeconds,
                     int threads = -1, int blockSize = -1, bool raceWarning = false)
{
    cout << "\n=== Run Summary ===" << endl;
    cout << "Algorithm  : " << algorithm << endl;
    cout << "Matrix     : " << n << "x" << n << endl;
    if (threads > 0)
        cout << "Threads    : " << threads << endl;
    if (blockSize > 0)
        cout << "Block size : " << blockSize << endl;

    double gflops = computeGflops(n, elapsedSeconds);
    cout << fixed << setprecision(6);
    cout << "Time (s)   : " << elapsedSeconds << endl;
    cout << setprecision(3);
    cout << "GFlop/s    : " << gflops << endl;

    // CSV line is easy to paste into report tables or spreadsheets.
    cout << "CSV        : "
         << algorithm << "," << n << ","
         << (threads > 0 ? to_string(threads) : "-") << ","
         << (blockSize > 0 ? to_string(blockSize) : "-") << ","
         << setprecision(6) << elapsedSeconds << ","
         << setprecision(3) << gflops << endl;

    if (raceWarning)
        cout << "Warning    : This variant has a race condition; values may be incorrect." << endl;
    cout << "===================\n" << endl;
}

// ─────────────────────────────────────────────────────────────
// PART 1 – Original functions (unchanged)
// ─────────────────────────────────────────────────────────────

void OnMult(int m_ar, int m_br) {
    double *pha, *phb, *phbt, *phc;
    pha  = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phb  = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phbt = (double *)malloc((m_ar * m_ar) * sizeof(double)); // transposed B
    phc  = (double *)malloc((m_ar * m_ar) * sizeof(double));

    for(int i = 0; i < m_ar; i++)
        for(int j = 0; j < m_ar; j++)
            pha[i*m_ar + j] = 1.0;
    for(int i = 0; i < m_br; i++)
        for(int j = 0; j < m_br; j++)
            phb[i*m_br + j] = (double)(i + 1);

        for(int i = 0; i < m_ar; i++)
        for(int j = 0; j < m_br; j++)
            phbt[j*m_ar + i] = phb[i*m_br + j];

    SYSTEMTIME Time1 = clock();

    for(int i = 0; i < m_ar; i++)
        for(int j = 0; j < m_br; j++) {
            double temp = 0;
            for(int k = 0; k < m_ar; k++)
                temp += pha[i*m_ar + k] * phbt[j*m_ar + k]; // both row access now
            phc[i*m_ar + j] = temp;
        }

    SYSTEMTIME Time2 = clock();
    double elapsed = (double)(Time2 - Time1) / CLOCKS_PER_SEC;
    printRunSummary("P1-V1-ijk(transposed B)", m_ar, elapsed);
    printResultPreview(phc, m_br);

    free(pha); free(phb); free(phbt); free(phc);
}


void OnMultLine(int m_ar, int m_br)
{
    SYSTEMTIME Time1, Time2;
    int i, j, k;

    double *pha, *phb, *phc;

    pha = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phb = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phc = (double *)malloc((m_ar * m_ar) * sizeof(double));

    for(i = 0; i < m_ar; i++)
        for(j = 0; j < m_ar; j++)
            pha[i*m_ar + j] = 1.0;

    for(i = 0; i < m_br; i++)
        for(j = 0; j < m_br; j++)
            phb[i*m_br + j] = (double)(i + 1);

    for(i = 0; i < m_ar; i++)
        for(j = 0; j < m_br; j++)
            phc[i*m_ar + j] = 0.0;

    Time1 = clock();

    for (i = 0; i < m_ar; i++) {
        int iRow = i * m_ar;
        for (k = 0; k < m_ar; k++) {
            double aik = pha[iRow + k];
            int kRow = k * m_br;
            for (j = 0; j < m_br; j++) {
                phc[iRow + j] += aik * phb[kRow + j];
            }
        }
    }

    Time2 = clock();
    double elapsed = (double)(Time2 - Time1) / CLOCKS_PER_SEC;
    printRunSummary("P1-V2-ikj(line)", m_ar, elapsed);
    printResultPreview(phc, m_br);

    free(pha);
    free(phb);
    free(phc);
}


void OnMultBlock(int m_ar, int m_br, int bkSize)
{
    SYSTEMTIME Time1, Time2;

    int i, j, k;
    int ii, jj, kk;

    double *pha, *phb, *phc;

    pha = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phb = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phc = (double *)malloc((m_ar * m_ar) * sizeof(double));

    for(i = 0; i < m_ar; i++)
        for(j = 0; j < m_ar; j++)
            pha[i*m_ar + j] = 1.0;

    for(i = 0; i < m_br; i++)
        for(j = 0; j < m_br; j++)
            phb[i*m_br + j] = (double)(i + 1);

    for(i = 0; i < m_ar; i++)
        for(j = 0; j < m_br; j++)
            phc[i*m_ar + j] = 0.0;

    Time1 = clock();

    for(ii = 0; ii < m_ar; ii += bkSize)
    {
        for(kk = 0; kk < m_ar; kk += bkSize)
        {
            for(jj = 0; jj < m_br; jj += bkSize)
            {
                for (i = ii; i < min(ii + bkSize, m_ar); i++)
                {
                    for(k = kk; k < min(kk + bkSize, m_ar); k++)
                    {
                        for(j = jj; j < min(jj + bkSize, m_br); j++)
                        {
                            double aik = pha[i*m_ar + k];
                            phc[i*m_ar + j] += aik * phb[k*m_br + j];
                        }
                    }
                }
            }
        }
    }

    Time2 = clock();
    double elapsed = (double)(Time2 - Time1) / CLOCKS_PER_SEC;
    printRunSummary("P1-V3-block", m_ar, elapsed, -1, bkSize);
    printResultPreview(phc, m_br);

    free(pha);
    free(phb);
    free(phc);
}


// ─────────────────────────────────────────────────────────────
// PART 2 – Parallel versions with OpenMP
// ─────────────────────────────────────────────────────────────

// Part 2.1 – Version 1 (ijk), Variant A
// #pragma omp parallel for on the outermost loop (correct approach)
void OnMultParallelV1A(int m_ar, int m_br, int numThreads)
{
    int i, j, k;
    double *pha, *phb, *phc;

    pha = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phb = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phc = (double *)malloc((m_ar * m_ar) * sizeof(double));

    for(i = 0; i < m_ar; i++)
        for(j = 0; j < m_ar; j++)
            pha[i*m_ar + j] = 1.0;

    for(i = 0; i < m_br; i++)
        for(j = 0; j < m_br; j++)
            phb[i*m_br + j] = (double)(i + 1);

    omp_set_num_threads(numThreads);

    double t1 = omp_get_wtime();

    #pragma omp parallel for
    for(i = 0; i < m_ar; i++)
        for(j = 0; j < m_br; j++) {
            double temp = 0;
            for(k = 0; k < m_ar; k++)
                temp += pha[i*m_ar + k] * phb[k*m_br + j];
            phc[i*m_ar + j] = temp;
        }

    double t2 = omp_get_wtime();
    double elapsed = t2 - t1;
    printRunSummary("P2-V1A-ijk-parallel-for", m_ar, elapsed, numThreads);
    printResultPreview(phc, m_br);

    free(pha); free(phb); free(phc);
}

// Part 2.1 – Version 1 (ijk), Variant B
// Manual inner-k partition across threads inside a parallel region
// NOTE: race condition on C[i][j] — included for analysis as per assignment
void OnMultParallelV1B(int m_ar, int m_br, int numThreads)
{
    int i, j, k;
    double *pha, *phb, *phc;

    pha = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phb = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phc = (double *)malloc((m_ar * m_ar) * sizeof(double));

    for(i = 0; i < m_ar; i++)
        for(j = 0; j < m_ar; j++)
            pha[i*m_ar + j] = 1.0;

    for(i = 0; i < m_br; i++)
        for(j = 0; j < m_br; j++)
            phb[i*m_br + j] = (double)(i + 1);

    for(i = 0; i < m_ar; i++)
        for(j = 0; j < m_br; j++)
            phc[i*m_ar + j] = 0.0;

    omp_set_num_threads(numThreads);

    double t1 = omp_get_wtime();

    // Deliberately bad strategy (for assignment analysis): split k across threads
    // and accumulate directly into C without synchronization, which causes races.
    // This preserves the intended "incorrect variant" behavior without the massive
    // per-(i,j) barrier overhead of omp-for inside nested loops.
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int kChunk = (m_ar + nth - 1) / nth;
        int kStart = tid * kChunk;
        int kEnd = min(kStart + kChunk, m_ar);

        for(i = 0; i < m_ar; i++)
            for(j = 0; j < m_br; j++) {
                double temp = 0;
                for(k = kStart; k < kEnd; k++)
                    temp += pha[i*m_ar + k] * phb[k*m_br + j];
                phc[i*m_ar + j] += temp; // intentional race condition
            }
    }

    double t2 = omp_get_wtime();
    double elapsed = t2 - t1;
    printRunSummary("P2-V1B-ijk-inner-k", m_ar, elapsed, numThreads, -1, true);
    printResultPreview(phc, m_br);

    free(pha); free(phb); free(phc);
}

// Part 2.1 – Version 2 (ikj), parallel outer loop
void OnMultParallelV2(int m_ar, int m_br, int numThreads)
{
    int i, j, k;
    double *pha, *phb, *phc;

    pha = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phb = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phc = (double *)malloc((m_ar * m_ar) * sizeof(double));

    for(i = 0; i < m_ar; i++)
        for(j = 0; j < m_ar; j++)
            pha[i*m_ar + j] = 1.0;

    for(i = 0; i < m_br; i++)
        for(j = 0; j < m_br; j++)
            phb[i*m_br + j] = (double)(i + 1);

    for(i = 0; i < m_ar; i++)
        for(j = 0; j < m_br; j++)
            phc[i*m_ar + j] = 0.0;

    omp_set_num_threads(numThreads);

    double t1 = omp_get_wtime();

    #pragma omp parallel for
    for(i = 0; i < m_ar; i++)
        for(k = 0; k < m_ar; k++)
            for(j = 0; j < m_br; j++)
                phc[i*m_ar + j] += pha[i*m_ar + k] * phb[k*m_br + j];

    double t2 = omp_get_wtime();
    double elapsed = t2 - t1;
    printRunSummary("P2-V2-ikj-parallel-for", m_ar, elapsed, numThreads);
    printResultPreview(phc, m_br);

    free(pha); free(phb); free(phc);
}

// ─────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────

int main(int argc, char *argv[])
{
    int lin, col, blockSize, numThreads;
    int op;

    do {
        cout << endl;
        cout << "── Part 1 ──────────────────────────" << endl;
        cout << "1. Multiplication" << endl;
        cout << "2. Line Multiplication" << endl;
        cout << "3. Block Multiplication" << endl;
        cout << "── Part 2 ──────────────────────────" << endl;
        cout << "4. Parallel V1-A  (ijk, omp parallel for outer)" << endl;
        cout << "5. Parallel V1-B  (ijk, manual inner-k split - race)" << endl;
        cout << "6. Parallel V2    (ikj, omp parallel for outer)" << endl;
        cout << "────────────────────────────────────" << endl;
        cout << "0. Exit" << endl;
        cout << "Selection?: ";
        cin >> op;

        if (op == 0) break;

        cout << "Dimensions: lins=cols ? ";
        cin >> lin;
        col = lin;

        switch (op) {
            case 1:
                OnMult(lin, col);
                break;
            case 2:
                OnMultLine(lin, col);
                break;
            case 3:
                cout << "Block Size? ";
                cin >> blockSize;
                OnMultBlock(lin, col, blockSize);
                break;
            case 4:
                cout << "Number of threads? ";
                cin >> numThreads;
                OnMultParallelV1A(lin, col, numThreads);
                break;
            case 5:
                cout << "Number of threads? ";
                cin >> numThreads;
                OnMultParallelV1B(lin, col, numThreads);
                break;
            case 6:
                cout << "Number of threads? ";
                cin >> numThreads;
                OnMultParallelV2(lin, col, numThreads);
                break;
        }

    } while (op != 0);

    return 0;
}