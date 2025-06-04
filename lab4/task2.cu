#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

#define N 131072  // 2^17

// ---------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------

bool is_sorted(float *arr, int n) {
    for (int i = 1; i < n; ++i)
        if (arr[i - 1] > arr[i]) return false;
    return true;
}

bool compare_arrays(float *a, float *b, int n, float epsilon = 1e-5) {
    for (int i = 0; i < n; ++i)
        if (fabs(a[i] - b[i]) > epsilon) return false;
    return true;
}

void generate_random_array(float *arr, int n) {
    for (int i = 0; i < n; ++i)
        arr[i] = (float)rand() / RAND_MAX * 1000.0f;
}

// ---------- ПОСЛЕДОВАТЕЛЬНАЯ MERGE SORT ----------

void merge(float *arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    float *L = (float*)malloc(n1 * sizeof(float));
    float *R = (float*)malloc(n2 * sizeof(float));

    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];

    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L);
    free(R);
}

void merge_sort(float *arr, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        merge_sort(arr, l, m);
        merge_sort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

// ---------- ПАРАЛЛЕЛЬНАЯ BITONIC SORT ----------

__global__ void bitonic_sort_kernel(float *data, int j, int k) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    unsigned int ixj = i ^ j;

    if (ixj > i) {
        if ((i & k) == 0) {
            if (data[i] > data[ixj]) {
                float tmp = data[i];
                data[i] = data[ixj];
                data[ixj] = tmp;
            }
        } else {
            if (data[i] < data[ixj]) {
                float tmp = data[i];
                data[i] = data[ixj];
                data[ixj] = tmp;
            }
        }
    }
}

void bitonic_sort(float *values) {
    float *dev_values;
    size_t size = N * sizeof(float);
    cudaMalloc((void**)&dev_values, size);
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

    int threads = 1024;
    int blocks = (N + threads - 1) / threads;

    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_kernel<<<blocks, threads>>>(dev_values, j, k);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_values);
}

// ---------- ГЛАВНАЯ ФУНКЦИЯ ----------

int main() {
    srand(time(NULL));
    float *arr_seq = (float*)malloc(N * sizeof(float));
    float *arr_par = (float*)malloc(N * sizeof(float));

    generate_random_array(arr_seq, N);
    memcpy(arr_par, arr_seq, N * sizeof(float));

    // Последовательная сортировка
    clock_t start_seq = clock();
    merge_sort(arr_seq, 0, N - 1);
    clock_t end_seq = clock();
    float time_seq = (float)(end_seq - start_seq) / CLOCKS_PER_SEC;

    // Параллельная сортировка
    clock_t start_par = clock();
    bitonic_sort(arr_par);
    clock_t end_par = clock();
    float time_par = (float)(end_par - start_par) / CLOCKS_PER_SEC;

    // Вывод результатов
    printf("Сортировка %d элементов...\n", N);
    printf("Sequential time: %.4f\n", time_seq);
    printf("Parallel time: %.4f\n", time_par);

    printf("%s Последовательная сортировка верна\n", is_sorted(arr_seq, N) ? "✅" : "❌");
    printf("%s Параллельная сортировка %s верна\n",
           is_sorted(arr_par, N) ? "✅" : "❌",
           compare_arrays(arr_seq, arr_par, N) ? "и совпадает с результатом" : "НО отличается");

    free(arr_seq);
    free(arr_par);
    return 0;
}
