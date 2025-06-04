#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

#define N 1000000

__global__ void sum_reduction_kernel(float* input, float* output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float x = 0;
    if (i < n)
        x = input[i];
    if (i + blockDim.x < n)
        x += input[i + blockDim.x];

    sdata[tid] = x;
    __syncthreads();

    // Редукция внутри блока
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

int main() {
    // Создание и заполнение массива
    float* h_array = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_array[i] = 1.0f;  // Для простоты сумма должна быть N
    }

    // sequential
    auto start_seq = std::chrono::high_resolution_clock::now();
    float sum_seq = 0.0f;
    for (int i = 0; i < N; i++) {
        sum_seq += h_array[i];
    }
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> seq_time = end_seq - start_seq;
    printf("Sequential time: %.3f ms\n", seq_time.count());

    // parallel
    float *d_input, *d_output;
    int blockSize = 256;
    int gridSize = (N + blockSize * 2 - 1) / (blockSize * 2);

    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, gridSize * sizeof(float));
    cudaMemcpy(d_input, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

    float* h_partial = (float*)malloc(gridSize * sizeof(float));
    float sum_parallel = 0.0f;

    auto start_par = std::chrono::high_resolution_clock::now();
    sum_reduction_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    auto end_par = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_partial, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < gridSize; i++) {
        sum_parallel += h_partial[i];
    }

    std::chrono::duration<double, std::milli> par_time = end_par - start_par;
    printf("Parallel time: %.3f ms\n", par_time.count());

    // Проверка результата
    printf("Sequential sum: %.2f\n", sum_seq);
    printf("Parallel sum: %.2f\n", sum_parallel);

    // Очистка
    free(h_array);
    free(h_partial);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
