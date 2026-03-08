import java.util.Scanner;

public class Mult {

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
        System.out.printf("Time: %.3f seconds%n", (time2 - time1) / 1e9);

        System.out.println("Result matrix:");
        for (int j = 0; j < Math.min(10, mBr); j++)
            System.out.print(phc[j] + " ");
        System.out.println();
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
        System.out.printf("Time: %.3f seconds%n", (time2 - time1) / 1e9);

        System.out.println("Result matrix:");
        for (int j = 0; j < Math.min(10, mBr); j++)
            System.out.print(phc[j] + " ");
        System.out.println();
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
        System.out.printf("Time: %.3f seconds%n", (time2 - time1) / 1e9);

        System.out.println("Result matrix:");
        for (int j = 0; j < Math.min(10, mBr); j++)
            System.out.print(phc[j] + " ");
        System.out.println();
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int op;

        do {
            System.out.println();
            System.out.println("1. Multiplication");
            System.out.println("2. Line Multiplication");
            System.out.println("3. Block Multiplication");
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
            }
        } while (op != 0);

        sc.close();
    }
}
