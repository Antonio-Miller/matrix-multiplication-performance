using System;
using System.Diagnostics;

class MatrixMultiplication
{
    void OnMult(int m_ar, int m_br)
    {
        Stopwatch timer = new Stopwatch();

        int i, j, k;
        double temp;

        double[] pha = new double[m_ar * m_ar];
        double[] phb = new double[m_ar * m_ar];
        double[] phc = new double[m_ar * m_ar];

        for (i = 0; i < m_ar; i++)
            for (j = 0; j < m_ar; j++)
                pha[i * m_ar + j] = 1.0;

        for (i = 0; i < m_br; i++)
            for (j = 0; j < m_br; j++)
                phb[i * m_br + j] = (double)(i + 1);

        timer.Start();

        for (i = 0; i < m_ar; i++)
        {
            for (j = 0; j < m_br; j++)
            {
                temp = 0;
                for (k = 0; k < m_ar; k++)
                {
                    temp += pha[i * m_ar + k] * phb[k * m_br + j];
                }
                phc[i * m_ar + j] = temp;
            }
        }

        timer.Stop();
        string st = string.Format("Time: {0:F3} seconds\n", timer.Elapsed.TotalSeconds);
        Console.Write(st);

        Console.WriteLine("Result matrix: ");
        for (i = 0; i < 1; i++)
        {
            for (j = 0; j < Math.Min(10, m_br); j++)
                Console.Write(phc[j] + " ");
        }
        Console.WriteLine();
    }

    // Line-by-line matrix multiplication
    void OnMultLine(int m_ar, int m_br)
    {

    }

    // Block matrix multiplication
    void OnMultBlock(int m_ar, int m_br, int bkSize)
    {

    }

    static void Main(string[] args)
    {
        MatrixMultiplication prog = new MatrixMultiplication();
        int lin, col, blockSize;
        int op;

        do
        {
            Console.WriteLine();
            Console.WriteLine("1. Multiplication");
            Console.WriteLine("2. Line Multiplication");
            Console.WriteLine("3. Block Multiplication");
            Console.WriteLine("0. Exit");
            Console.Write("Selection?: ");
            op = int.Parse(Console.ReadLine());

            if (op == 0)
                break;

            Console.Write("Dimensions: lins=cols ? ");
            lin = int.Parse(Console.ReadLine());
            col = lin;

            switch (op)
            {
                case 1:
                    prog.OnMult(lin, col);
                    break;
                case 2:
                    prog.OnMultLine(lin, col);
                    break;
                case 3:
                    Console.Write("Block Size? ");
                    blockSize = int.Parse(Console.ReadLine());
                    prog.OnMultBlock(lin, col, blockSize);
                    break;
            }

        } while (op != 0);
    }
}
