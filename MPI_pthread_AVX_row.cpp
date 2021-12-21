#include <iostream>
#include <windows.h>
#include <pthread.h>
#include <windows.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2
#include <semaphore.h>
#include <mpi.h>
#include <stdint.h>

using namespace std;

#define n 20
#define thread_count 4

float A[n][n];
int id[thread_count];
long long head, tail , freq;
sem_t sem_parent;//���߳�
pthread_barrier_t childbarrier_row;
pthread_barrier_t childbarrier_col;

struct data
{
    int id;
    int begin;
    int end;
    int myid;
}datagroups[thread_count];


void init()
{
    //初始化上三角矩阵
    for(int i=0;i<n;i++)
        for(int j=i;j<n;j++)
            A[i][j]=i+j+2;

    for(int i=1;i<n;i++)
        for(int j=0;j<n;j++)
            A[i][j]=A[i][j]+A[0][j];
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            A[j][i]=A[j][i]+A[j][0];
}
void printA()
{
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
            cout<<A[i][j]<<" ";
        cout<<endl;
    }
}
void normal_gausseliminate()
{
    for(int k=0;k<n;k++)
    {

        for(int j=k+1;j<n;j++)
        {
            A[k][j]=A[k][j]/A[k][k];
        }
        A[k][k]=1;
        for(int i=k+1;i<n;i++)
        {
            for(int j=k+1;j<n;j++)
            {
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
            A[i][k]=0;
        }
    }
}
void * dealwithbyrow_AVX(void * datai)
{
    data* datagroup= (data*)datai;
    __m256 t1,t2,t3;
    for(int k=0;k<n;k++)
    {
        int begin=datagroup->begin +datagroup->id*((datagroup->end-datagroup->begin)/thread_count);
        int end=begin+(datagroup->end-datagroup->begin)/thread_count;
        if(datagroup->id==thread_count-1)
            end=datagroup->end;
        int preprocessnumber=(n-k-1)%8;
        int begincol=k+1+preprocessnumber;

        for(int i=(begin>=(k+1)?begin:k+1);i<end;i++)
        {
            for(int j=k+1;j<begincol;j++)
            {
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
            A[i][k]=0;
        }
        
        for(int i=(begin>=(k+1)?begin:(k+1));i<end;i++)
        {
            float head1[8]={A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k]};
            t3=_mm256_loadu_ps(head1);
            for(int j=begincol;j<n;j+=8)
            {
                t1=_mm256_loadu_ps(A[k]+j);
                t2=_mm256_loadu_ps(A[i]+j);
                t1=_mm256_mul_ps(t1,t3);
                t2=_mm256_sub_ps(t2,t1);
                _mm256_storeu_ps(A[i]+j,t2);
            }
            A[i][k]=0;
        }
        sem_post(&sem_parent);
        pthread_barrier_wait(&childbarrier_row);
    }
    pthread_exit(NULL);
}




int main(int argc,char* argv[]){
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    pthread_barrier_init(&childbarrier_row, NULL,thread_count+1);
    pthread_barrier_init(&childbarrier_col,NULL, thread_count+1);
    sem_init(&sem_parent, 0, 0);
    pthread_t threadID[thread_count];

    int myid, numprocs;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    int distributerow=n/(numprocs-1);
    //0号进程首先完成初始化的工作，再将按行划分的每一行传给不同的进程
    if(myid==0)
    {
        init();//初始化
        //任务分发
        for(int i=1;i<numprocs;i++)//从1号开始分发
        {
            int begin=(i-1)*distributerow;
            int end=begin+distributerow;
            if(i==numprocs-1)
                end=n;
            int count=(end-begin)*n;//发送数据个数
            //从begin行开始传
            MPI_Send((void *)A[begin],count,MPI_FLOAT,i,0,MPI_COMM_WORLD);
        }
        printA();
    }
    else//接受消息后并更新对应的矩阵
    {
        int begin=(myid-1)*distributerow;
        int end=begin+distributerow;
        if(myid==numprocs-1)
            end=n;
        int count=(end-begin)*n;
        MPI_Recv((void *)A[begin],count,MPI_FLOAT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }

    //开始进行消去
    int begin=(myid-1)*distributerow;
    int end=begin+distributerow;
    if(myid==numprocs-1)
        end=n;
    int count=(end-begin)*n;
    for(int k=0;k<n;k++)
    {
        if(myid==0)
        {
            if(k!=0)
            {
                int source=(k/distributerow+1)<(numprocs-1)?(k/distributerow+1):(numprocs-1);
                MPI_Recv((void *)(A[k]+k), n-k, MPI_FLOAT, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            __m256 t1,t2,t3;
            int preprocessnumber=(n-k-1)%8;
            int begincol=k+1+preprocessnumber;
            float head[8]={A[k][k],A[k][k],A[k][k],A[k][k],A[k][k],A[k][k],A[k][k],A[k][k]};
            t2=_mm256_loadu_ps(head);
            for(int j=k+1;j<k+1+preprocessnumber;j++)
            {
                A[k][j]=A[k][j]/A[k][k];
            }
            for(int j=begincol;j<n;j+=8)
            {
                //A[k][j]=A[k][j]/A[k][k];
                t1=_mm256_loadu_ps(A[k]+j);
                t1=_mm256_div_ps(t1,t2);
                _mm256_storeu_ps(A[k]+j,t1);
            }
            A[k][k]=1;
            // for(int j=i+1;j<n;j++)
            //     A[i][j]=A[i][j]/A[i][i];
            // //将更新完的行传给剩余的进程
            // A[i][i]=1; 
            
                    
        }
        //数据广播，不能放到if语句，采用规约树广播，此时共享
        MPI_Bcast((void *)(A[k]+k),n-k,MPI_FLOAT,0,MPI_COMM_WORLD);

        //收到数据后才能跑
        if(myid!=0)
        {
            // for(int j=(begin>(k+1)?begin:k+1);j<end;j++)//注意begin与i+1的关系
            // {
            //     for(int i=k+1;i<n;i++)
            //     {
            //         A[j][i]=A[j][i]-A[j][k]*A[k][i];
            //     }
            //     A[j][k]=0;
            // } 

            //将begin与end的数据均匀的分给线程的数量

            if(k==0)
            {
                for(int i=0;i<thread_count;i++)
                {
                    datagroups[i].id=i;
                    datagroups[i].begin=begin>(k+1)?begin:k+1;
                    datagroups[i].end=end;
                    datagroups[i].myid=myid;
                    pthread_create(&threadID[i],NULL,dealwithbyrow_AVX,(void*)&datagroups[i]);
                }
            }
            else
                pthread_barrier_wait(&childbarrier_row);
            for(int i=0;i<thread_count;i++)//
            {
                sem_wait(&sem_parent);
            }
            if((k+1<n)&&(k+1)>=begin&&(k+1)<end)//更新的数据传回0号
            {
                MPI_Send((void *)(A[k+1]+k+1), n-(k+1), MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
            }
        }
    }
    if(myid!=0)
    {
        pthread_barrier_wait(&childbarrier_row);
        for(int i=0;i<thread_count;i++)
        {
            pthread_join(threadID[i],NULL);
        }
        //将每个进程更新后的结果传回
        MPI_Send((void *)A[begin],count,MPI_FLOAT,0,1,MPI_COMM_WORLD);
    }   
    else
    {
        for(int i=1;i<numprocs;i++)
        {
            int begin=(i-1)*distributerow;
            int end=begin+distributerow;
            if(i==numprocs-1)
                end=n;
            int count=(end-begin)*n;//发送数据个数
            MPI_Recv((void *)A[begin],count,MPI_FLOAT,i,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        } 
    }
    if(myid==0)
            printA();
    MPI_Finalize();

}
