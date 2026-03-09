#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <cstdlib>
#include <omp.h>

using namespace std;

#define SYSTEMTIME clock_t

// ─────────────────────────────────────────────────────────────
// PART 1 – Original functions (unchanged)
// ─────────────────────────────────────────────────────────────

void OnMult(int m_ar, int m_br) 
{
    SYSTEMTIME Time1, Time2;
    
    char st[100];
    double temp;
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

    Time1 = clock();

    for(i = 0; i < m_ar; i++)
    {
        for(j = 0; j < m_br; j++)
        {
            temp = 0;
            for(k = 0; k < m_ar; k++)
            {    
                temp += pha[i*m_ar + k] * phb[k*m_br + j];
            }
            phc[i*m_ar + j] = temp;
        }
    }

    Time2 = clock();
    snprintf(st, sizeof(st), "Time: %3.3f seconds\n",
         (double)(Time2 - Time1) / CLOCKS_PER_SEC);
    cout << st;

    cout << "Result matrix: " << endl;
    for(i = 0; i < 1; i++)
    {
        for(j = 0; j < min(10, m_br); j++)
            cout << phc[j] << " ";
    }
    cout << endl;

    free(pha);
    free(phb);
    free(phc);
}


void OnMultLine(int m_ar, int m_br)
{
    SYSTEMTIME Time1, Time2;
    char st[100];
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

    for(i = 0; i < m_ar; i++){
        for(k = 0; k < m_ar; k++){
            for(j = 0; j < m_br; j++){
                phc[i*m_ar + j] += pha[i*m_ar + k] * phb[k*m_br + j];
            }
        }
    }

    Time2 = clock();
    snprintf(st, sizeof(st), "Time: %3.3f seconds\n",
         (double)(Time2 - Time1) / CLOCKS_PER_SEC);
    cout << st;

    cout << "Result matrix: " << endl;
    for(i = 0; i < 1; i++)
    {
        for(j = 0; j < min(10, m_br); j++)
            cout << phc[j] << " ";
    }
    cout << endl;

    free(pha);
    free(phb);
    free(phc);
}


void OnMultBlock(int m_ar, int m_br, int bkSize)
{
    SYSTEMTIME Time1, Time2;

    char st[100];
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
                            phc[i*m_ar + j] += pha[i*m_ar + k] * phb[k*m_br + j];
                        }
                    }
                }
            }
        }
    }

    Time2 = clock();
    snprintf(st, sizeof(st), "Time: %3.3f seconds\n",
         (double)(Time2 - Time1) / CLOCKS_PER_SEC);
    cout << st;

    cout << "Result matrix: " << endl;
    for(i = 0; i < 1; i++)
    {
        for(j = 0; j < min(10, m_br); j++)
            cout << phc[j] << " ";
    }
    cout << endl;

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
    double gflops = 2.0 * m_ar * m_ar * m_ar / (elapsed * 1e9);
    printf("[V1-ParallelFor] Threads: %d | Time: %.3f s | GFlop/s: %.3f\n",
           numThreads, elapsed, gflops);

    cout << "Result matrix: " << endl;
    for(j = 0; j < min(10, m_br); j++)
        cout << phc[j] << " ";
    cout << endl;

    free(pha); free(phb); free(phc);
}

// Part 2.1 – Version 1 (ijk), Variant B
// #pragma omp parallel + #pragma omp for on the innermost k loop
// NOTE: race condition on temp and C[i][j] — included for analysis as per assignment
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

    omp_set_num_threads(numThreads);

    double t1 = omp_get_wtime();

    #pragma omp parallel
    for(i = 0; i < m_ar; i++)
        for(j = 0; j < m_br; j++) {
            double temp = 0;
            #pragma omp for
            for(k = 0; k < m_ar; k++)
                temp += pha[i*m_ar + k] * phb[k*m_br + j];
            phc[i*m_ar + j] = temp;
        }

    double t2 = omp_get_wtime();
    double elapsed = t2 - t1;
    double gflops = 2.0 * m_ar * m_ar * m_ar / (elapsed * 1e9);
    printf("[V1-ParallelInner] Threads: %d | Time: %.3f s | GFlop/s: %.3f\n",
           numThreads, elapsed, gflops);

    cout << "Result matrix: " << endl;
    for(j = 0; j < min(10, m_br); j++)
        cout << phc[j] << " ";
    cout << endl;

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
    double gflops = 2.0 * m_ar * m_ar * m_ar / (elapsed * 1e9);
    printf("[V2-ParallelFor] Threads: %d | Time: %.3f s | GFlop/s: %.3f\n",
           numThreads, elapsed, gflops);

    cout << "Result matrix: " << endl;
    for(j = 0; j < min(10, m_br); j++)
        cout << phc[j] << " ";
    cout << endl;

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
        cout << "5. Parallel V1-B  (ijk, omp parallel + omp for inner k)" << endl;
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