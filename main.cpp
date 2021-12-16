#include<iostream>
#include<math.h>
#include<cstdlib>
#include<windows.h>
#include<stdlib.h>
#include<xmmintrin.h>
#include<fstream>
#include<immintrin.h>
#include<pthread.h>


// 定义线程数
const int NUM_THREADS = 2;
const int DATASZ = 1 << 12;

using namespace std;

// 定义结构，表示线程的ID
typedef struct {
	int threatId;
}threadParm_t;

// 信号量，用于给timer上锁
pthread_mutex_t	mutex;
pthread_barrier_t barrier;

const int maxn = DATASZ + 10;
long long head, freq;
float A[maxn][maxn];
float B[maxn][maxn];

int n;

void print_matrix(int n, float matrix[][maxn]) {

	for (int i = 0; i < n;i++) {
		for (int j = 0; j < n; j++) {
			cout << matrix[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

void rebuild(int n, float matrix[][maxn]) {

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < i; j++) {

			matrix[i][j] = 0;
		}
		matrix[i][i] = 1;
	}

}

// 无任何优化的LU分解算法
void LU_line(int n) {

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
}

// 方式1 按行进行划分，不使用sse
void* LU_pthread(void *parm) {

	threadParm_t *p = (threadParm_t *)parm;
	int r = p->threatId;
	long long tail;


	for (int k = 0; k < n;k++) {

		for (int i = k + 1; i < n; i++) {

			if ((i % NUM_THREADS) == r) {

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


// 方式二 按列进行划分，不使用sse
void *LU_pthread_matrix(void *parm) {


	threadParm_t *p = (threadParm_t *)parm;
	int r = p->threatId;
	long long tail;

	for (int k = 0; k < n; k++) {
		for (int j = k + 1 + r;j < n;j += NUM_THREADS) {
			B[k][j] = B[k][j] / A[k][k];
		}
		B[k][k] = 1;

		pthread_barrier_wait(&barrier);

		for (int i = k + 1 + r; i < n; i += NUM_THREADS) {
			for (int j = k + 1;j < n;j++) {
				B[i][j] = B[i][j] - B[i][k] * B[k][j];
			}
			B[i][k] = 0;
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

// 方式三 对按行进行划分，使用sse
void* LU_pthread_sse(void *parm) {

	threadParm_t *p = (threadParm_t *)parm;
	int r = p->threatId;
	long long tail;
	__m128 t1, t2, t3;

	for (int k = 0; k < n;k++) {

		for (int i = k + 1; i < n; i++) {

			if ((i % NUM_THREADS) == r) {

				B[i][k] = B[i][k] / B[k][k];

				int offset = (n - k - 1) % 4;
				for (int j = k + 1; j < k+1+offset; j++) {
					B[i][j] = B[i][j] - B[i][k] * B[k][j];
				}
				t2 = _mm_set_ps(B[i][k], B[i][k], B[i][k], B[i][k]);
				for (int j = k + 1 + offset ; j<n; j+=4) {
					t3 = _mm_loadu_ps(B[k] + j);
					t1 = _mm_loadu_ps(B[i] + j);
					t2 = _mm_mul_ps(t2, t3);
					t1 = _mm_sub_ps(t1, t2);
					_mm_storeu_ps(B[i] + j, t1);
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


// 方式四 按行进行划分，使用avx
void* LU_pthread_avx(void *parm) {

	threadParm_t *p = (threadParm_t *)parm;
	int r = p->threatId;
	long long tail;
	__m256  t1, t2, t3;

	for (int k = 0; k < n;k++) {

		for (int i = k + 1; i < n; i++) {

			if ((i % NUM_THREADS) == r) {

				B[i][k] = B[i][k] / B[k][k];

				int offset = (n - k - 1) % 8;
				for (int j = k + 1; j < k + 1 + offset; j++) {
					B[i][j] = B[i][j] - B[i][k] * B[k][j];
				}
				t2 = _mm256_set_ps(B[i][k], B[i][k], B[i][k], B[i][k], B[i][k], B[i][k], B[i][k], B[i][k]);
				for (int j = k + 1 + offset; j<n; j += 8) {
					t3 = _mm256_loadu_ps(B[k] + j);
					t1 = _mm256_loadu_ps(B[i] + j);
					t2 = _mm256_mul_ps(t2, t3);
					t1 = _mm256_sub_ps(t1, t2);
					_mm256_storeu_ps(B[i] + j, t1);
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


void init() {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			A[i][j] = i + j + 1 + rand() * 100;
			B[i][j] = A[i][j];
		}
	}
}

void init_test() {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i < j) {
				A[i][j] = i + 1;
				B[i][j] = A[i][j];
			}
			else {
				A[i][j] = j + 1;
				B[i][j] = A[i][j];
			}
		}
	}
}

int main_main(int datasize) {

	double time1 = 0;
	//n = 8;
	//init_test();

	//print_matrix(n, A);

	long long tail;
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

	// similar to CLOCKS_PER_SEC
	//QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
	//LU_line(n);
	//QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
	//time1 = (tail - head) * 1000.0 / freq;

	//cout << "single thread: " << time1 << "ms" << endl;


	/*-----------------多线程的函数的测试*--------------------*/

	pthread_t thread[NUM_THREADS];
	threadParm_t threadParm[NUM_THREADS];

	mutex = PTHREAD_MUTEX_INITIALIZER;
	pthread_barrier_init(&barrier, NULL, NUM_THREADS);
    int i;
    /*
	QueryPerformanceCounter((LARGE_INTEGER*)&head); // start timer
	for (i = 0; i < NUM_THREADS; i++) {
		threadParm[i].threatId = i;
		pthread_create(&thread[i], NULL, LU_pthread, (void *)&threadParm[i]);
	}

	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_join(thread[i], 0);
	}

	pthread_mutex_destroy(&mutex);*/

	//cout << "正确性测试：" << endl <<"the next matix is A:" << endl;
	//print_matrix(n, A);

	//cout << "the next matix is B:" << endl;
	//rebuild(n, B);
	//print_matrix(n, B);

	/*-------------------------------------------------------------------------*/
	/*以上为测试多线程程序的正确性*/

	n = datasize;
	cout << "-----------------------------------------------" << endl;
	cout << "矩阵规模:" << n << endl;
	cout << "线程数： " << NUM_THREADS << endl;
	//------------------------朴素算法---------------------------
	init();

	// similar to CLOCKS_PER_SEC
	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
	LU_line(n);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
	time1 = (tail - head) * 1000.0 / freq;

	cout << "Test: Serial algorithm" << endl << "Thread 1: " << time1 << "ms" << endl << endl;
    //-----------------------------------------------------------
	//----------------------pthread按列划分----------------------
	cout << "Test: Pthreads divide by column" << endl;
	init();
	mutex = PTHREAD_MUTEX_INITIALIZER;
	pthread_barrier_init(&barrier, NULL, NUM_THREADS);

	QueryPerformanceCounter((LARGE_INTEGER*)&head); // start timer
	for (i = 0; i < NUM_THREADS; i++) {
		threadParm[i].threatId = i;
		pthread_create(&thread[i], NULL, LU_pthread_matrix, (void *)&threadParm[i]);
	}

	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_join(thread[i], 0);
	}

	pthread_mutex_destroy(&mutex);
    cout << endl;
    //------------------------------------------------------------
	//---------------------pthread按行划分------------------------
	cout << "Test: Pthreads divide by row" << endl;
	init();
	mutex = PTHREAD_MUTEX_INITIALIZER;
	pthread_barrier_init(&barrier, NULL, NUM_THREADS);

	QueryPerformanceCounter((LARGE_INTEGER*)&head); // start timer
	for (i = 0; i < NUM_THREADS; i++) {
		threadParm[i].threatId = i;
		pthread_create(&thread[i], NULL, LU_pthread, (void *)&threadParm[i]);
	}

	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_join(thread[i], 0);
	}

	pthread_mutex_destroy(&mutex);
    cout << endl;
    //------------------------------------------------------------
	//-------------------SSE+Pthread按行划分算法测试----------------------
	cout << "Test: SSE + Pthreads divide by row" << endl;
	init();
	mutex = PTHREAD_MUTEX_INITIALIZER;
	pthread_barrier_init(&barrier, NULL, NUM_THREADS);
	QueryPerformanceCounter((LARGE_INTEGER*)&head); // start timer
	for (i = 0; i < NUM_THREADS; i++) {
		threadParm[i].threatId = i;
		pthread_create(&thread[i], NULL, LU_pthread_sse, (void *)&threadParm[i]);
	}

	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_join(thread[i], 0);
	}

	pthread_mutex_destroy(&mutex);
	cout << endl;
    //------------------------------------------------------------
	//-------------------SSE+Pthread按行划分算法测试----------------------
	cout << "Test: AVX + Pthreads divide by row" << endl;
	init();
	mutex = PTHREAD_MUTEX_INITIALIZER;
	pthread_barrier_init(&barrier, NULL, NUM_THREADS);
	QueryPerformanceCounter((LARGE_INTEGER*)&head); // start timer
	for (i = 0; i < NUM_THREADS; i++) {
		threadParm[i].threatId = i;
		pthread_create(&thread[i], NULL, LU_pthread_avx, (void *)&threadParm[i]);
	}

	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_join(thread[i], 0);
	}

	pthread_mutex_destroy(&mutex);
	cout << endl;
    //------------------------------------------------------------
	//system("pause");
}
int main(){
    for(int i = 1 << 9; i <= DATASZ; i <<= 1){
        main_main(i);
    }
    //main_main(4096);
}
