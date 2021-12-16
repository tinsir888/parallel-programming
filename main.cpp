#include <iostream>
#include <time.h>
#include <windows.h>
#include <math.h>
#include <cstdio>
#include <pmmintrin.h>
//#include <avxintrin.h>
#include <immintrin.h>
using namespace std;

int N = 0;

void output(float **matrix) {
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            printf("%.2f ", matrix[i][j]);
        }
        puts("");
    }
    return;
}

void ord(float **matrix) {
	for (int k = 0; k < N; k++) {
        //optimizable part one
		float tmp = matrix[k][k];
		matrix[k][k] = 1.0;
		for (int j = k + 1; j < N; j++) {
			matrix[k][j] /= tmp;
		}
        //optimizable part two
		for (int i = k + 1; i < N; i++) {
			float tmp = matrix[i][k];
			for (int j = k + 1; j < N; j++) {
				matrix[i][j] -= tmp * matrix[k][j];
			}
			matrix[i][k] = 0;
		}
	}
	//output(matrix);
}

void SSE_optimize_only_first(float **matrix){
	__m128 t1, t2, t3, t4;
	for (int k = 0; k < N; k++) {
        //PART ONE
		float tmp[4] = {matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
		t1 = _mm_loadu_ps(tmp);
		for (int j = N - 4; j >= k; j -= 4) {
			t2 = _mm_loadu_ps(matrix[k] + j);
			t3 = _mm_div_ps(t2, t1);
			_mm_storeu_ps(matrix[k] + j, t3);
		}

		if (k & 3 != (N & 3)) {
			for (int j = k; (j & 3) != (N & 3); j++) {
				matrix[k][j] /= tmp[0];
			}
		}

		for (int j = (N & 3) - 1; j >= 0; j--) {
			matrix[k][j] /= tmp[0];
		}
        //PART TWO
        for (int i = k + 1; i < N; i++) {
			float tmp = matrix[i][k];
			for (int j = k + 1; j < N; j++) {
				matrix[i][j] -= tmp * matrix[k][j];
			}
			matrix[i][k] = 0;
		}
	}
}

void SSE_optimize_only_second(float **matrix){
	__m128 t1, t2, t3, t4;
	for (int k = 0; k < N; k++) {
        float tmp = matrix[k][k];
		matrix[k][k] = 1.0;
		for (int j = k + 1; j < N; j++) {
			matrix[k][j] /= tmp;
		}
        //PART TWO
		for (int i = k + 1; i < N; i++) {
			float tmp[4] = {matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]};
			t1 = _mm_loadu_ps(tmp);
			for (int j = N - 4; j > k; j -= 4) {
				t2 = _mm_loadu_ps(matrix[i] + j);
				t3 = _mm_loadu_ps(matrix[k] + j);
				t4 = _mm_sub_ps(t2, _mm_mul_ps(t1, t3));
				_mm_storeu_ps(matrix[i] + j, t4);
			}

			for (int j = k + 1; (j & 3) != (N & 3); j++) {
				matrix[i][j] -= matrix[i][k] * matrix[k][j];
			}
			matrix[i][k] = 0;
		}
	}
}
void SSE_optimize_both_align(float **matrix){
	__m128 t1, t2, t3, t4;
	int mod;
	float temp;
	for(int k = 0; k < N; k ++){
        t1 = _mm_set1_ps(matrix[k][k]);
        mod = 4 - (k & 3);
        temp = matrix[k][k];
        for(int j = k; j < k + mod; j ++)
            matrix[k][j] /= temp;
        for(int j = k + mod; j < N; j += 4){
            t2 = _mm_load_ps(matrix[k] + j);
            t2 = _mm_div_ps(t2, t1);
            _mm_store_ps(matrix[k] + j, t2);
        }
        for(int i = k + 1; i < N; i++){
            t1 = _mm_set1_ps(matrix[i][k]);
            temp = matrix[i][k];
            for(int j = k + 1; j < k + mod; j ++){
                matrix[i][j] -= temp * matrix[k][j];
            }
            for(int j = k + mod; j < N; j += 4){
				t2 = _mm_load_ps(matrix[i] + j);
				t3 = _mm_load_ps(matrix[k] + j);
				t4 = _mm_sub_ps(t2, _mm_mul_ps(t1, t3));
				_mm_store_ps(matrix[i] + j, t4);
            }
            matrix[i][k] = 0;
        }
	}
	//output(matrix);
}
void SSE_optimize_both(float **matrix) {
	__m128 t1, t2, t3, t4;
	for (int k = 0; k < N; k++) {
        //PART ONE
		float tmp[4] = {matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
		t1 = _mm_loadu_ps(tmp);
		for (int j = N - 4; j >= k; j -= 4) {
			t2 = _mm_loadu_ps(matrix[k] + j);
			t3 = _mm_div_ps(t2, t1);
			_mm_storeu_ps(matrix[k] + j, t3);
		}

		if (k & 3 != (N & 3)) {
			for (int j = k; (j & 3) != (N & 3); j++) {
				matrix[k][j] /= tmp[0];
			}
		}

		for (int j = (N & 3) - 1; j >= 0; j--) {
			matrix[k][j] /= tmp[0];
		}
        //PART TWO
		for (int i = k + 1; i < N; i++) {
			float tmp[4] = {matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]};
			t1 = _mm_loadu_ps(tmp);
			for (int j = N - 4; j > k; j -= 4) {
				t2 = _mm_loadu_ps(matrix[i] + j);
				t3 = _mm_loadu_ps(matrix[k] + j);
				t4 = _mm_sub_ps(t2, _mm_mul_ps(t1, t3));
				_mm_storeu_ps(matrix[i] + j, t4);
			}

			for (int j = k + 1; (j & 3) != (N & 3); j++) {
				matrix[i][j] -= matrix[i][k] * matrix[k][j];
			}
			matrix[i][k] = 0;
		}
	}
}

void AVX_optimize_both(float **matrix){
    __m256 t1,t2,t3,t4;
	for (int k = 0; k < N; k++) {
        //PART ONE
		float tmp[8] = {matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k],
		matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
		t1 = _mm256_loadu_ps(tmp);
		for (int j = N - 8; j >= k; j -= 8) {
			t2 = _mm256_loadu_ps(matrix[k] + j);
			t3 = _mm256_div_ps(t2, t1);
			_mm256_storeu_ps(matrix[k] + j, t3);
		}

		if (k & 7 != (N & 7)) {
			for (int j = k; (j & 7) != (N & 7); j++) {
				matrix[k][j] /= tmp[0];
			}
		}

		for (int j = (N & 7) - 1; j >= 0; j--) {
			matrix[k][j] /= tmp[0];
		}
        //PART TWO
		for (int i = k + 1; i < N; i++) {
			float tmp[8] = {matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k],
			matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]};
			t1 = _mm256_loadu_ps(tmp);
			for (int j = N - 8; j > k; j -= 8) {
				t2 = _mm256_loadu_ps(matrix[i] + j);
				t3 = _mm256_loadu_ps(matrix[k] + j);
				t4 = _mm256_sub_ps(t2, _mm256_mul_ps(t1, t3));
				_mm256_storeu_ps(matrix[i] + j, t4);
			}

			for (int j = k + 1; (j & 7) != (N & 7); j++) {
				matrix[i][j] -= matrix[i][k] * matrix[k][j];
			}
			matrix[i][k] = 0;
		}
	}
}

int main() {
    //freopen("res.txt","w",stdout);
	srand((int)time(0));
	for (int n = 1<<5; n <= 1<<11; n <<= 1) {
		cout << "N = " << n << ":" << endl;
		N = n;
		float **matrix1 = new float*[N];//ordinary algorithm
		float **matrix2 = new float*[N];//optimize both parts
		float **matrix3 = new float*[N];//optimize only first part
		float **matrix4 = new float*[N];//optimize only second part
		float **matrix5 = new float*[N];//optimize both parts with AVX256
		float **matrix6 = new float*[N];//optimize both parts with aligned strategy
		for (int i = 0; i < N; i++) {
			matrix1[i] = new float[N];
			matrix2[i] = new float[N];
			matrix3[i] = new float[N];
			matrix4[i] = new float[N];
			matrix5[i] = new float[N];
			matrix6[i] = new float[N];
		}
		//随机数据生成
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
                float tmp = rand() % 10000;
				matrix6[i][j] = matrix5[i][j] = matrix4[i][j] = matrix3[i][j] = matrix2[i][j] = matrix1[i][j] = 1.00 * tmp / 100.00;
			}
		}


		long long head, tail, freq;
		double t1, t2, t3, t4, t5, t6;

		QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

		QueryPerformanceCounter((LARGE_INTEGER *)&head);
		ord(matrix1);
		QueryPerformanceCounter((LARGE_INTEGER *)&tail);
		t1 = (double)(tail - head) * 1000 / freq;
		cout << "无优化的串行算法:" << t1 << "ms" << endl;

		QueryPerformanceCounter((LARGE_INTEGER *)&head);
		SSE_optimize_only_first(matrix2);
		QueryPerformanceCounter((LARGE_INTEGER *)&tail);
		t2 = (double)(tail - head) * 1000 / freq;
		cout << "采用SSE并行只优化第一部分的串行算法:" << t2 << "ms" << endl;
        cout << "采用SSE并行只优化第一部分对比无优化性能提升率:" << (t1 - t2) / t1 * 100 << "%" << endl << endl;

		QueryPerformanceCounter((LARGE_INTEGER *)&head);
		SSE_optimize_only_first(matrix3);
		QueryPerformanceCounter((LARGE_INTEGER *)&tail);
		t3 = (double)(tail - head) * 1000 / freq;
		cout << "采用SSE并行只优化第二部分的串行算法:" << t3 << "ms" << endl;
        cout << "采用SSE并行只优化第二部分对比无优化性能提升率:" << (t1 - t3) / t1 * 100 << "%" << endl << endl;

		QueryPerformanceCounter((LARGE_INTEGER *)&head);
		SSE_optimize_both(matrix4);
		QueryPerformanceCounter((LARGE_INTEGER *)&tail);
		t4 = (double)(tail - head) * 1000 / freq;
		cout << "采用SSE并行同时优化两部分的算法:" << t4 << "ms" << endl;
        cout << "采用SSE并行同时优化两部分对比无优化性能提升率:" << (t1 - t4) / t1 * 100 << "%" << endl << endl;

		QueryPerformanceCounter((LARGE_INTEGER *)&head);
		AVX_optimize_both(matrix5);
		QueryPerformanceCounter((LARGE_INTEGER *)&tail);
		t5 = (double)(tail - head) * 1000 / freq;
		cout << "采用AVX256并行同时优化两部分的算法:" << t5 << "ms" << endl;
        cout << "采用AVX256并行同时优化两部分对比无优化性能提升率:" << (t1 - t5) / t1 * 100 << "%" << endl;
        cout << "采用AVX256并行同时优化两部分对比采用SSE性能提升率:"  << (t4 - t5) / t4 * 100 << "%" << endl << endl;

		QueryPerformanceCounter((LARGE_INTEGER *)&head);
		SSE_optimize_both_align(matrix6);
		QueryPerformanceCounter((LARGE_INTEGER *)&tail);
		t6 = (double)(tail - head) * 1000 / freq;
		cout << "采用SSE并行采用对齐策略同时优化两部分的算法:" << t6 << "ms" << endl;
        cout << "采用SSE并行采用对齐策略同时优化对比无优化性能提升率:" << (t1 - t6) / t1 * 100 << "%" << endl;
        cout << "采用SSE并行采用对齐策略同时优化对比采用SSE不对齐策略性能提升率:"  << (t4 - t6) / t4 * 100 << "%" << endl << endl;

        printf("12加速比: %.2f\n", t1 / t2);
        printf("13加速比: %.2f\n", t1 / t3);
        printf("24加速比: %.2f\n", t2 / t4);
        printf("34加速比: %.2f\n", t3 / t4);
        printf("14加速比: %.2f\n", t1 / t4);
        printf("15加速比: %.2f\n", t1 / t5);
        printf("45加速比: %.2f\n", t4 / t5);
        printf("16加速比: %.2f\n", t1 / t6);
        printf("46加速比: %.2f\n\n\n\n", t4 / t6);
        //output(matrix2);//test
	}

	return 0;
}
