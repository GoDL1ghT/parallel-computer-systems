#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "CUDA Error: %s:%d, code: %d, reason: %s\n", __FILE__, __LINE__, error, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void array_operations_kernel(const double* a, const double* b, double* sum,
                                       double* diff, double* prod, double* quot, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        sum[idx] = a[idx] + b[idx];
        diff[idx] = a[idx] - b[idx];
        prod[idx] = a[idx] * b[idx];
        quot[idx] = (b[idx] != 0.0) ? a[idx] / b[idx] : 0.0;
    }
}

void array_operations_sequential(const double* a, const double* b, double* sum,
                                 double* diff, double* prod, double* quot, int N) {
    for (int i = 0; i < N; ++i) {
        sum[i] = a[i] + b[i];
        diff[i] = a[i] - b[i];
        prod[i] = a[i] * b[i];
        quot[i] = (b[i] != 0.0) ? a[i] / b[i] : 0.0;
    }
}

int main(int argc, char* argv[]) {
    int N = 1000000; // по умолчанию миллион элементов

    if (argc > 1) {
        N = atoi(argv[1]);
        if (N < 100000) {
            printf("Размер массива должен быть не менее 100000. Установлено %d.\n", N);
        }
    }

    printf("Array size: %d\n", N);

    double *a = (double*)malloc(N * sizeof(double));
    double *b = (double*)malloc(N * sizeof(double));

    double *sum_seq = (double*)malloc(N * sizeof(double));
    double *diff_seq = (double*)malloc(N * sizeof(double));
    double *prod_seq = (double*)malloc(N * sizeof(double));
    double *quot_seq = (double*)malloc(N * sizeof(double));

    double *sum_par = (double*)malloc(N * sizeof(double));
    double *diff_par = (double*)malloc(N * sizeof(double));
    double *prod_par = (double*)malloc(N * sizeof(double));
    double *quot_par = (double*)malloc(N * sizeof(double));

    if (!a || !b || !sum_seq || !diff_seq || !prod_seq || !quot_seq ||
        !sum_par || !diff_par || !prod_par || !quot_par) {
        fprintf(stderr, "Ошибка выделения памяти\n");
        return -1;
    }

    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        a[i] = ((double)rand() / RAND_MAX) * 100.0 + 1.0;
        b[i] = ((double)rand() / RAND_MAX) * 100.0 + 1.0;
    }

    // Последовательный запуск и измерение времени
    clock_t start_seq = clock();
    array_operations_sequential(a, b, sum_seq, diff_seq, prod_seq, quot_seq, N);
    clock_t end_seq = clock();
    double seq_time = (double)(end_seq - start_seq) / CLOCKS_PER_SEC;

    // Параллельный запуск на CUDA
    double *d_a, *d_b, *d_sum, *d_diff, *d_prod, *d_quot;
    CHECK(cudaMalloc((void**)&d_a, N * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_b, N * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_sum, N * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_diff, N * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_prod, N * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_quot, N * sizeof(double)));

    CHECK(cudaMemcpy(d_a, a, N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start_event, stop_event;
    CHECK(cudaEventCreate(&start_event));
    CHECK(cudaEventCreate(&stop_event));

    CHECK(cudaEventRecord(start_event));
    array_operations_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_sum, d_diff, d_prod, d_quot, N);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop_event));

    CHECK(cudaEventSynchronize(stop_event));

    float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));

    // Копируем результаты обратно, если нужно (но не обязательно, раз не сравниваем)
    CHECK(cudaMemcpy(sum_par, d_sum, N * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(diff_par, d_diff, N * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(prod_par, d_prod, N * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(quot_par, d_quot, N * sizeof(double), cudaMemcpyDeviceToHost));

    printf("Sequential sort time: %.6f seconds\n", seq_time);
    printf("Parallel sort time: %.6f seconds\n", milliseconds / 1000.0f);

    free(a); free(b);
    free(sum_seq); free(diff_seq); free(prod_seq); free(quot_seq);
    free(sum_par); free(diff_par); free(prod_par); free(quot_par);

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_sum));
    CHECK(cudaFree(d_diff));
    CHECK(cudaFree(d_prod));
    CHECK(cudaFree(d_quot));

    CHECK(cudaEventDestroy(start_event));
    CHECK(cudaEventDestroy(stop_event));

    return 0;
}
