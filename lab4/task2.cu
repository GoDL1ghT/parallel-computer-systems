#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <math.h>

#define N (1 << 17)  // 131072 элементов > 100000


// Последовательный Merge Sort

void merge(float *array, int left, int mid, int right, float *temp) {
    int i = left, j = mid, k = left;
    while (i < mid && j < right) {
        if (array[i] < array[j])
            temp[k++] = array[i++];
        else
            temp[k++] = array[j++];
    }
    while (i < mid) temp[k++] = array[i++];
    while (j < right) temp[k++] = array[j++];
    for (i = left; i < right; ++i)
        array[i] = temp[i];
}

void merge_sort_seq(float *array, int left, int right, float *temp) {
    if (right - left <= 1) return;
    int mid = (left + right) / 2;
    merge_sort_seq(array, left, mid, temp);
    merge_sort_seq(array, mid, right, temp);
    merge(array, left, mid, right, temp);
}


// Параллельный Bitonic Sort на CUDA

__global__ void bitonic_sort_step(float *dev_values, int j, int k) {
    unsigned int i, ixj;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i ^ j;

    if (ixj > i) {
        if ((i & k) == 0) {
            if (dev_values[i] > dev_values[ixj]) {
                float temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        } else {
            if (dev_values[i] < dev_values[ixj]) {
                float temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
    }
}

void bitonic_sort(float *values) {
    float *dev_values;
    size_t size = N * sizeof(float);
    cudaMalloc((void**)&dev_values, size);
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

    dim3 blocks(N / 1024);
    dim3 threads(1024);

    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_values);
}

int is_sorted(float *arr, int size) {
    for (int i = 0; i < size - 1; ++i)
        if (arr[i] > arr[i + 1]) return 0;
    return 1;
}

int compare_arrays(float *a, float *b, int size) {
    for (int i = 0; i < size; ++i)
        if (fabs(a[i] - b[i]) > 1e-5)
            return 0;
    return 1;
}


int main() {
    float *arr_seq = (float*)malloc(sizeof(float) * N);
    float *arr_par = (float*)malloc(sizeof(float) * N);
    float *temp = (float*)malloc(sizeof(float) * N);

    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        float val = (float)(rand() % 100000) / 100.0f;
        arr_seq[i] = val;
        arr_par[i] = val;
    }

    printf("Сортировка %d элементов...\n", N);

    // Последовательная сортировка
    clock_t start_seq = clock();
    merge_sort_seq(arr_seq, 0, N, temp);
    clock_t end_seq = clock();
    printf("Sequential time: %.4f\n", (double)(end_seq - start_seq) / CLOCKS_PER_SEC);

    // Параллельная сортировка
    clock_t start_par = clock();
    bitonic_sort(arr_par);
    clock_t end_par = clock();
    printf("Parallel time: %.4f\n", (double)(end_par - start_par) / CLOCKS_PER_SEC);

    // Проверка
    if (!is_sorted(arr_seq, N)) printf("❌ Последовательная сортировка НЕ верна\n");
    else printf("✅ Последовательная сортировка верна\n");

    if (!is_sorted(arr_par, N)) printf("❌ Параллельная сортировка НЕ верна\n");
    else printf("✅ Параллельная сортировка верна\n");

    if (compare_arrays(arr_seq, arr_par, N))
        printf("✅ Результаты сортировок совпадают\n");
    else
        printf("❌ Результаты сортировок не совпадают\n");

    free(arr_seq);
    free(arr_par);
    free(temp);
    return 0;
}
