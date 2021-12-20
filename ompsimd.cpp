#include <iostream>
#include<math.h>
#include<cstdlib>
#include<stdlib.h>
#include<xmmintrin.h>
#include<fstream>
#include<immintrin.h>
#include<pthread.h>
#include<sys/time.h>
#include<omp.h>
#include<time.h>
#include<windows.h>
// 定义线程数
#define THREAD_NUM 12

using namespace std;

// 定义结构，表示线程的ID
typedef struct {
	int threatId;
}threadParm_t;

// 信号量，用于给timer上锁
pthread_mutex_t	mutex;
pthread_barrier_t barrier;

struct timeval t1, t2;    // timers
const int maxn = 1 << 13;
long long head, freq;
float A[maxn][maxn];
float B[maxn][maxn];

int n;

void print_matrix(int n, float** matrix) {

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << matrix[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

void rebuild(int n, float** matrix) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < i; j++) {
			matrix[i][j] = 0;
		}
		matrix[i][i] = 1;
	}
}

void init() {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			A[i][j] = i + j + 1 + rand() * 100;
			B[i][j] = A[i][j];
		}
	}
}

/*void init_test() {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i < j) {
				A[i][j] = i + 1;
				B[i][j] = A[i][j];
			} else {
				A[i][j] = j + 1;
				B[i][j] = A[i][j];
			}
		}
	}
}*/

void* LU_pthread(void *parm) {

	threadParm_t *p = (threadParm_t *)parm;
	int r = p->threatId;
	long long tail; 
	for (int k = 0; k < n;k++) {
		for (int i = k + 1; i < n; i++) {
			if ((i % THREAD_NUM) == r) {
				B[i][k] = B[i][k] / B[k][k];
				for (int j = k + 1; j<n; j++) {
					B[i][j] = B[i][j] - B[i][k] * B[k][j];
				}
			}
		}
		pthread_barrier_wait(&barrier);
	}
	pthread_mutex_lock(&mutex);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	cout << "Thread " << r << ": " << (tail - head) * 1000.0 / freq << "ms" << endl;
	pthread_mutex_unlock(&mutex);

	pthread_exit(0);

	return 0;
}

void main_main(int nn) {
	n = nn;
	//init_test();
	int k;
	init();
	cout << "线程数:" << THREAD_NUM << endl;
//按行进行划分，结合SIMD-AVX 
	gettimeofday(&t1, NULL);
	#pragma omp parallel num_threads(THREAD_NUM) shared(k)
	{
		#pragma omp barrier
		int r = omp_get_thread_num();
		__m256 t1_1, t2, t3;
		for (int k = 0; k < n; k++){
			for(int i = k + 1; i < n; i++){
				if(i%THREAD_NUM == r){
					B[i][k] = B[i][k] / B[k][k];
					int offset = (n - k - 1) % 8;
					for(int j = k + 1; j < k + 1 + offset; j ++){
						B[i][j] = B[i][j] - B[i][k] * B[k][j];
					}
					t2 = _mm256_set_ps(B[i][k], B[i][k], B[i][k], B[i][k],
					B[i][k], B[i][k], B[i][k], B[i][k]);
					for (int j = k + 1 + offset ; j < n; j += 8) {
						t3 = _mm256_load_ps(B[k] + j);
						t1_1 = _mm256_load_ps(B[i] + j);
						t2 = _mm256_mul_ps(t2, t3);
						t1_1 = _mm256_sub_ps(t1_1, t2);
						_mm256_store_ps(B[i] + j, t1_1);
					}
				}
			}
		}
	}
	gettimeofday(&t2, NULL);
	printf("OMP按行划分with AVX256 : %.3fms.\n", ((t2.tv_sec - t1.tv_sec) * 1e6 + t2.tv_usec - t1.tv_usec) / 1e3);
//按行进行划分，结合SIMD-SSE 
	init();
	gettimeofday(&t1, NULL); // start timer
	#pragma omp parallel num_threads(THREAD_NUM) shared(k)
	{
		#pragma omp barrier
		int r = omp_get_thread_num();
		__m128 t1_1, t2, t3;
		for (int k = 0; k < n; k++) {
			for (int i = k + 1; i < n; i++) {
				if ((i % THREAD_NUM) == r) {
					B[i][k] = B[i][k] / B[k][k];
					int offset = (n - k - 1) % 4;
					for (int j = k + 1; j < k + 1 + offset; j++) {
						B[i][j] = B[i][j] - B[i][k] * B[k][j];
					}
					t2 = _mm_set_ps(B[i][k], B[i][k], B[i][k], B[i][k]);
					for (int j = k + 1 + offset ; j < n; j += 4) {
						t3 = _mm_load_ps(B[k] + j);
						t1_1 = _mm_load_ps(B[i] + j);
						t2 = _mm_mul_ps(t2, t3);
						t1_1 = _mm_sub_ps(t1_1, t2);
						_mm_store_ps(B[i] + j, t1_1);
					}

				}
			}
		}
	}
	gettimeofday(&t2, NULL);
	printf("OMP按行划分with SSE128 : %.3fms.\n", ((t2.tv_sec - t1.tv_sec) * 1e6 + t2.tv_usec - t1.tv_usec) / 1e3);
//按行进行划分，不结合SIMD 
	init();
	gettimeofday(&t1, NULL);
	#pragma omp parallel num_threads(THREAD_NUM) shared(k)
	{
		#pragma omp barrier
		int r = omp_get_thread_num();
		for (int k = 0; k < n; k++) {
			for (int i = k + 1; i < n; i++) {
				if ((i % THREAD_NUM) == r) {
					B[i][k] = B[i][k] / B[k][k];
					for (int j = k + 1; j < n; j++) {
						B[i][j] = B[i][j] - B[i][k] * B[k][j];
					}
				}
			}
		}
	}
	gettimeofday(&t2, NULL);
	printf("OMP按行划分without SIMD : %.3fms.\n", ((t2.tv_sec - t1.tv_sec) * 1e6 + t2.tv_usec - t1.tv_usec) / 1e3);
//Pthread按行划分
	cout << "Pthread按行划分without SIMD :" << endl;
	init();
	pthread_t thread[THREAD_NUM];
	threadParm_t threadParm[THREAD_NUM];
	mutex = PTHREAD_MUTEX_INITIALIZER;
	pthread_barrier_init(&barrier, NULL, THREAD_NUM);
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head); // start timer
	for (int i = 0; i < THREAD_NUM; i++) {
		threadParm[i].threatId = i;
		pthread_create(&thread[i], NULL, LU_pthread, (void *)&threadParm[i]);
	}

	for (int i = 0; i < THREAD_NUM; i++) {
		pthread_join(thread[i], 0);
	}

	pthread_mutex_destroy(&mutex);
    //cout << endl; 
//朴素算法
	init();
	gettimeofday(&t1, NULL);
	for (int k = 0; k < n;k++) {
		for (int j = k + 1; j < n; j++) {
			A[k][j] = A[k][j] / A[k][k];
		}
		A[k][k] = 1.0;
		for (int i = k + 1; i < n; i++) {
			for (int l = k + 1; l<n; l++) {
				A[i][l] = A[i][l] - A[i][k] * A[k][l];
			}
			A[i][k] = 0;
		}
	}
	gettimeofday(&t2, NULL);
	printf("朴素算法 : %.3fms.\n", ((t2.tv_sec - t1.tv_sec) * 1e6 + t2.tv_usec - t1.tv_usec) / 1e3);
}
int main(){
	freopen("12thread.txt","w",stdout);
	for(int i=7;i<=11;i++){
		cout<<"n="<<(int)(1<<i)<<endl;
		main_main(1<<i);
		cout<<endl;
	}
}
