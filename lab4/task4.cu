#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define N 1000  // Размерность массива (1000x1000 = 1_000_000 элементов)

// === CUDA Kernel ===
__global__ void kernelOps(float *a, float *b, float *add, float *sub, float *mul, float *div) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        add[idx] = a[idx] + b[idx];
        sub[idx] = a[idx] - b[idx];
        mul[idx] = a[idx] * b[idx];
        div[idx] = a[idx] / b[idx];
    }
}

// === Хост-класс ===
struct ArrayOps {
    float *a, *b;
    float *res_add_seq, *res_sub_seq, *res_mul_seq, *res_div_seq;
    float *res_add_gpu, *res_sub_gpu, *res_mul_gpu, *res_div_gpu;

    void fillArrays() {
        for (int i = 0; i < N * N; i++) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(i % 100 + 1); // избежать деления на 0
        }
    }

    void sequentialOps() {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < N * N; i++) {
            res_add_seq[i] = a[i] + b[i];
            res_sub_seq[i] = a[i] - b[i];
            res_mul_seq[i] = a[i] * b[i];
            res_div_seq[i] = a[i] / b[i];
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Sequential time: " << elapsed.count() << " ms\n";
    }

    void parallelOps() {
        float *d_a, *d_b, *d_add, *d_sub, *d_mul, *d_div;

        cudaMalloc(&d_a, N * N * sizeof(float));
        cudaMalloc(&d_b, N * N * sizeof(float));
        cudaMalloc(&d_add, N * N * sizeof(float));
        cudaMalloc(&d_sub, N * N * sizeof(float));
        cudaMalloc(&d_mul, N * N * sizeof(float));
        cudaMalloc(&d_div, N * N * sizeof(float));

        cudaMemcpy(d_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, N * N * sizeof(float), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        int threadsPerBlock = 256;
        int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;
        kernelOps<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_add, d_sub, d_mul, d_div);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        std::cout << "Parallel time: " << ms << " ms\n";

        cudaMemcpy(res_add_gpu, d_add, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(res_sub_gpu, d_sub, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(res_mul_gpu, d_mul, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(res_div_gpu, d_div, N * N * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_add);
        cudaFree(d_sub);
        cudaFree(d_mul);
        cudaFree(d_div);
    }

    void run() {
        a = new float[N * N];
        b = new float[N * N];
        res_add_seq = new float[N * N];
        res_sub_seq = new float[N * N];
        res_mul_seq = new float[N * N];
        res_div_seq = new float[N * N];
        res_add_gpu = new float[N * N];
        res_sub_gpu = new float[N * N];
        res_mul_gpu = new float[N * N];
        res_div_gpu = new float[N * N];

        fillArrays();
        sequentialOps();
        parallelOps();

        delete[] a;
        delete[] b;
        delete[] res_add_seq;
        delete[] res_sub_seq;
        delete[] res_mul_seq;
        delete[] res_div_seq;
        delete[] res_add_gpu;
        delete[] res_sub_gpu;
        delete[] res_mul_gpu;
        delete[] res_div_gpu;
    }
};

int main() {
    ArrayOps ops;
    ops.run();
    return 0;
}
