#include <iostream>
#include<pmmintrin.h>
#include<time.h>
#include<windows.h>
#include<math.h>

using namespace std;

int N = 0;

float** normal_gauss(float **matrix) {
	for (int k = 0; k < N; k++) {
		float tmp = matrix[k][k];
		for (int j = k; j < N; j++) {
			matrix[k][j] = matrix[k][j] / tmp;
		}
		for (int i = k + 1; i < N; i++) {
			float tmp2 = matrix[i][k];
			for (int j = k + 1; j < N; j++) {
				matrix[i][j] = matrix[i][j] - tmp2 * matrix[k][j];
			}
			matrix[i][k] = 0;
		}
	}
	return matrix;
}


void SSE_gauss(float **matrix) {
	__m128 t1, t2, t3, t4;
	for (int k = 0; k < N; k++) {
		float tmp[4] = {matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
		t1 = _mm_loadu_ps(tmp);
		for (int j = N - 4; j >= k; j -= 4) {
			t2 = _mm_loadu_ps(matrix[k] + j);
			t3 = _mm_div_ps(t2, t1);
			_mm_storeu_ps(matrix[k] + j, t3);
		}

		if (k % 4 != (N % 4)) {
			for (int j = k; j % 4 != (N % 4); j++) {
				matrix[k][j] = matrix[k][j] / tmp[0];
			}
		}

		for (int j = (N % 4) - 1; j >= 0; j--) {
			matrix[k][j] = matrix[k][j] / tmp[0];
		}

		for (int i = k + 1; i < N; i++) {
			float tmp[4] = {matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]};
			t1 = _mm_loadu_ps(tmp);
			for (int j = N - 4; j > k; j -= 4) {
				t2 = _mm_loadu_ps(matrix[i] + j);
				t3 = _mm_loadu_ps(matrix[k] + j);
				t4 = _mm_sub_ps(t2, _mm_mul_ps(t1, t3));
				_mm_storeu_ps(matrix[i] + j, t4);
			}
			for (int j = k + 1; j % 4 != (N % 4); j++) {
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			}
			matrix[i][k] = 0;
		}
	}
}


int main() {
	srand((int)time(0));
	int times = 3;
	for (int n = 8; n <= 8192; n = pow(2, times)) {
		times += 1;
		cout << "When N = " << n << ":" << endl;
		N = n;
		float **matrix1 = new float*[N];
		float **matrix2 = new float*[N];
		for (int i = 0; i < N; i++) {
			matrix1[i] = new float[N];
			matrix2[i] = new float[N];
		}
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				matrix1[i][j] = rand() % 100;
				matrix2[i][j] = rand() % 100;
			}
		}


		long long head, tail, freq;
		double t1, t2;

		QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

		QueryPerformanceCounter((LARGE_INTEGER *)&head);
		float **X = normal_gauss(matrix1);
		QueryPerformanceCounter((LARGE_INTEGER *)&tail);
		cout << "normal_gauss:" << (double)(tail - head) / freq << "s" << endl;
		t1 = (double)(tail - head) / freq;

		QueryPerformanceCounter((LARGE_INTEGER *)&head);
		SSE_gauss(matrix2);
		QueryPerformanceCounter((LARGE_INTEGER *)&tail);
		cout << "SSE_gauss:" << (double)(tail - head) / freq << "s" << endl;
		t2 = (double)(tail - head) / freq;

		cout << "性能提升率:" << (t1 - t2) / t1 * 100 << "%" << endl << endl;
	}

	return 0;
}
