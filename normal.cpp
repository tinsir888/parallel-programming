#include <iostream>
#include <mpi.h>
#include <stdint.h>

using namespace std;

#define n 10
#define thread_count 4

float A[n][n];



void init()
{
    //初始化上三角矩阵
    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++)
            A[i][j] = i + j + 2;

    for (int i = 1; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = A[i][j] + A[0][j];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[j][i] = A[j][i] + A[j][0];
}
void printA()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            cout << A[i][j] << " ";
        cout << endl;
    }
}
void normal_gausseliminate()
{
    for (int k = 0; k < n; k++)
    {

        for (int j = k + 1; j < n; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}





int main(int argc, char *argv[])
{
    int myid, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    int distributerow = n / (numprocs - 1);
    //0号进程首先完成初始化的工作，再将按行划分的每一行传给不同的进程
    if (myid == 0)
    {
        init(); //初始化
        //任务分发
        for (int i = 1; i < numprocs; i++) //从1号开始分发
        {
            int begin = (i - 1) * distributerow;
            int end = begin + distributerow;
            if (i == numprocs - 1)
                end = n;
            int count = (end - begin) * n; //发送数据个数
            //从begin行开始传
            MPI_Send((void *)A[begin], count, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }
        //printA();
    }
    else //接受消息后并更新对应的矩阵
    {
        int begin = (myid - 1) * distributerow;
        int end = begin + distributerow;
        if (myid == numprocs - 1)
            end = n;
        int count = (end - begin) * n;
        MPI_Recv((void *)A[begin], count, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if(myid==1)
        printA();
    //开始进行消去
    int begin = (myid - 1) * distributerow;
    int end = begin + distributerow;
    if (myid == numprocs - 1)
        end = n;
    int count = (end - begin) * n;

    for (int k = 0; k < n; k++)
    {
        if (myid == 0)
        {
            if(k!=0)
            {
                int source=(k/distributerow+1)<(numprocs-1)?(k/distributerow+1):(numprocs-1);
                MPI_Recv((void *)(A[k]+k), n-k, MPI_FLOAT, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            for (int j = k + 1; j < n; j++)
                A[k][j] = A[k][j] / A[k][k];
            //将更新完的行传给剩余的进程
            A[k][k] = 1;
        }
        //数据广播，不能放到if语句，采用规约树广播，此时共享
        MPI_Bcast((void *)(A[k] + k), n - k, MPI_FLOAT, 0, MPI_COMM_WORLD);

        //收到数据后才能跑
        if (myid != 0)
        {
            for (int j = (begin > (k + 1) ? begin : k + 1); j < end; j++) //注意begin与i+1的关系
            {
                for (int i = k + 1; i < n; i++)
                {
                    A[j][i] = A[j][i] - A[j][k] * A[k][i];
                }
                A[j][k] = 0;
            }
            if((k+1<n)&&(k+1)>=begin&&(k+1)<end)//更新的数据传回0号
            {
                MPI_Send((void *)(A[k+1]+k+1), n-(k+1), MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
            }
        }
        
    }
    if (myid != 0)
    {
        MPI_Send((void *)A[begin], count, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }
    else
    {
        for (int i = 1; i < numprocs; i++)
        {
            int begin = (i - 1) * distributerow;
            int end = begin + distributerow;
            if (i == numprocs - 1)
                end = n;
            int count = (end - begin) * n; //发送数据个数
            MPI_Recv((void *)A[begin], count, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    if (myid == 0)
        printA();
    MPI_Finalize();
}
