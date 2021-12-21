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

#define n 10
#define thread_count 1

float A[n][n];
int id[thread_count];
long long head, tail , freq;
sem_t sem_parent;
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
void * dealwithbyrow_SSE(void * datai)
{
    data* datagroup= (data*)datai;
    __m128 t1,t2,t3;
    for(int k=0;k<n;k++)
    {
        
        int begin=datagroup->begin +datagroup->id*((datagroup->end-datagroup->begin)/thread_count);
        int end=begin+(datagroup->end-datagroup->begin)/thread_count;
        if(datagroup->id==thread_count-1)
            end=datagroup->end;
        int preprocessnumber=(n-k-1)%4;
        int begincol=k+1+preprocessnumber;

        for(int i=(begin>=(k+1)?begin:k+1);i<end;i++)
        {
            for(int j=k+1;j<n;j++)
            {
                    A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }

            A[i][k]=0;
        }

        for(int i=(begin>=(k+1)?begin:(k+1));i<end;i++)
        {
            float head1[4]={A[i][k],A[i][k],A[i][k],A[i][k]};
            t3=_mm_loadu_ps(head1);
            for(int j=begincol;j<n;j+=4)
            {
                t1=_mm_loadu_ps(A[k]+j);
                t2=_mm_loadu_ps(A[i]+j);
                t1=_mm_mul_ps(t1,t3);
                t2=_mm_sub_ps(t2,t1);
                _mm_store_ss(A[i]+j,t2);
            }
            A[i][k]=0;
        }
        sem_post(&sem_parent);
        pthread_barrier_wait(&childbarrier_row);
    }
    pthread_exit(NULL);
}


void * dealwithbycol_SSE(void * datai)
{
    data* datagroup= (data*)datai;
    __m128 t1,t2,t3;
    
    for(int k=0;k<n;k++)
    {
        int begin=datagroup->begin +datagroup->id*((datagroup->end-datagroup->begin)/thread_count);
        int end=begin+(datagroup->end-datagroup->begin)/thread_count;
        if(datagroup->id==thread_count-1)
            end=datagroup->end;
        int preprocessnumber=(n-k-1)%4;
        int begincol=k+1+preprocessnumber;

        // if(k==0&&datagroup->myid==1)
        //     printA();
        for(int j=begin>=(k+1)?begin:(k+1);j<end;j++)
        {       
            A[k][j]=A[k][j]/A[k][k];
        }
        // if(k==0&&datagroup->myid==1)
        //     printA();
        for(int j=k+1;j<n;j++)
        {
            for(int i=(begin>=(k+1)?begin:(k+1));i<end;i++)
            {
                A[j][i]=A[j][i]-A[j][k]*A[k][i];
            }
        }
        // if(k==1&&datagroup->myid==1)
        //     printA();
        sem_post(&sem_parent);
        pthread_barrier_wait(&childbarrier_col);

    }
    pthread_exit(NULL);
}

void Gausseliminate_pthread_col_SSE()
{
    pthread_t threadID[thread_count];
    for(int k=0;k<n;k++)
    {
        if(k==0)
        {
            for(int i=0;i<thread_count;i++)
            {
                pthread_create(&threadID[i],NULL,dealwithbycol_SSE,(void*)&id[i]);
            }
        }
        for(int i=0;i<thread_count;i++)
        {
            sem_wait(&sem_parent);
        }
        pthread_barrier_wait(&childbarrier_col);
        A[k][k]=1;
        for(int i=k+1;i<n;i++)
            A[i][k]=0;


    }
    for(int i=0;i<thread_count;i++)
    {
        pthread_join(threadID[i],NULL);
    }
    return;
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
    //int distributerow=n/(numprocs-1);
    int distributecol=n/(numprocs);
    //0号进程首先完成初始化的工作，再将按行划分的每一行传给不同的进程
    if(myid==0)
    {
        init();//初始化
        //自定义数据类型
        for(int i=1;i< numprocs;i++)
        {
            int begin=i*distributecol;
            int end=begin+distributecol;
            if(i==numprocs-1)
                end=n;
            MPI_Datatype block;
            MPI_Type_vector(n,(end-begin),n,MPI_FLOAT,&block);
            MPI_Type_commit(&block);
            MPI_Send((void *)(A[0]+begin),1,block,i,0,MPI_COMM_WORLD);
        }
        //printA();
    }
    else//接受消息后并更新对应的矩阵
    {
        int begin=myid*distributecol;
        int end=begin+distributecol;
        if(myid==numprocs-1)
            end=n;
        MPI_Datatype block;
        MPI_Type_vector(n,(end-begin),n,MPI_FLOAT,&block);
        MPI_Type_commit(&block);
        MPI_Recv((void *)(A[0]+begin),1,block,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }

    //开始进行消去
    int begin=(myid)*distributecol;
    int end=begin+distributecol;
    if(myid==numprocs-1)
        end=n;
    for(int k=0;k<n;k++)
    {
        int source=k/distributecol;//注意source
        if(source>=numprocs)
            source=numprocs-1;
        MPI_Datatype temcol;
        MPI_Type_vector(n-k,1,n,MPI_FLOAT,&temcol);
        MPI_Type_commit(&temcol);
        // if(k==1&&myid==1)
        //     printA();   
        MPI_Bcast((void *)(A[k]+k),1,temcol,source,MPI_COMM_WORLD);//将用于矩阵更新的列传出
        // if(k==0&&myid==1)
        //     printA();   
        if(k==0)
        {
            for(int i=0;i<thread_count;i++)
            {
                datagroups[i].id=i;
                datagroups[i].begin=begin;
                datagroups[i].end=end;
                datagroups[i].myid=myid;
                pthread_create(&threadID[i],NULL,dealwithbycol_SSE,(void*)&datagroups[i]);
            }
        }
        else
        {
            pthread_barrier_wait(&childbarrier_col);
        }
        for(int i=0;i<thread_count;i++)
        {
            sem_wait(&sem_parent);
        }
        A[k][k]=1;
        for(int i=k+1;i<n;i++)
            A[i][k]=0;   
    }
    pthread_barrier_wait(&childbarrier_col);
    for(int i=0;i<thread_count;i++)
    {
        pthread_join(threadID[i],NULL);
    }


    if(myid!=0)
    {
        //将每个进程更新后的结果传回
        //MPI_Send((void *)A[begin],count,MPI_FLOAT,0,1,MPI_COMM_WORLD);
        MPI_Datatype block;
        MPI_Type_vector(n,(end-begin),n,MPI_FLOAT,&block);
        MPI_Type_commit(&block);
        MPI_Send((void *)(A[0]+begin),1,block,0,1,MPI_COMM_WORLD);
    }   
    else
    {
        for(int i=1;i<numprocs;i++)
        {
            int begin=i*distributecol;
            int end=begin+distributecol;
            if(i==numprocs-1)
                end=n;
            MPI_Datatype block;
            MPI_Type_vector(n,(end-begin),n,MPI_FLOAT,&block);
            MPI_Type_commit(&block);
            MPI_Recv((void *)(A[0]+begin),1,block,i,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        } 
    }
    if(myid==0)
        printA();

    MPI_Finalize();

}
