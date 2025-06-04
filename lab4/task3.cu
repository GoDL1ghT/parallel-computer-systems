#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <time.h>

#define CHECK(call)                                                             \
    do {                                                                       \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__);       \
            fprintf(stderr, "code: %d, reason: %s\n",                         \
                   error, cudaGetErrorString(error));                          \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while(0)

// Constants
#define DEFAULT_ARRAY_SIZE 1000000
#define DEFAULT_CHUNK_SIZE 1000000
#define DEFAULT_THREADS_PER_BLOCK 256
#define MIN_VALUE 1.0
#define MAX_VALUE 101.0

__global__ void array_operations_kernel_cuda(const double* a, const double* b, 
                                            double* sum, double* diff, 
                                            double* prod, double* quot, 
                                            int N_chunk) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_chunk) {
        sum[idx] = a[idx] + b[idx];
        diff[idx] = a[idx] - b[idx];
        prod[idx] = a[idx] * b[idx];
        quot[idx] = (b[idx] != 0.0) ? a[idx] / b[idx] : 0.0;
    }
}

void array_operations_sequential(const double* a, const double* b, 
                                double* sum, double* diff, 
                                double* prod, double* quot, 
                                int N_total) {
    for (int i = 0; i < N_total; ++i) {
        sum[i] = a[i] + b[i];
        diff[i] = a[i] - b[i];
        prod[i] = a[i] * b[i];
        quot[i] = (b[i] != 0.0) ? a[i] / b[i] : 0.0;
    }
}

void process_in_chunks_cuda(const double* a, const double* b, 
                           double* sum, double* diff, 
                           double* prod, double* quot, 
                           int N_total, int chunk_size, 
                           int threadsPerBlock) {
    double *d_a, *d_b, *d_sum, *d_diff, *d_prod, *d_quot;

    // Allocate device memory
    CHECK(cudaMalloc((void**)&d_a, chunk_size * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_b, chunk_size * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_sum, chunk_size * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_diff, chunk_size * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_prod, chunk_size * sizeof(double)));
    CHECK(cudaMalloc((void**)&d_quot, chunk_size * sizeof(double)));

    const int blocksPerGrid = (chunk_size + threadsPerBlock - 1) / threadsPerBlock;

    // Process array in chunks
    for (int i = 0; i < N_total; i += chunk_size) {
        const int current_chunk_size = (i + chunk_size > N_total) ? (N_total - i) : chunk_size;

        // Copy data to device
        CHECK(cudaMemcpy(d_a, a + i, current_chunk_size * sizeof(double), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_b, b + i, current_chunk_size * sizeof(double), cudaMemcpyHostToDevice));

        // Launch kernel
        array_operations_kernel_cuda<<<blocksPerGrid, threadsPerBlock>>>
            (d_a, d_b, d_sum, d_diff, d_prod, d_quot, current_chunk_size);
        CHECK(cudaGetLastError());

        // Copy results back to host
        CHECK(cudaMemcpy(sum + i, d_sum, current_chunk_size * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(diff + i, d_diff, current_chunk_size * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(prod + i, d_prod, current_chunk_size * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(quot + i, d_quot, current_chunk_size * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Cleanup
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_sum));
    CHECK(cudaFree(d_diff));
    CHECK(cudaFree(d_prod));
    CHECK(cudaFree(d_quot));
}

void initialize_arrays(double* a, double* b, int N) {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        a[i] = MIN_VALUE + (double)rand() / RAND_MAX * (MAX_VALUE - MIN_VALUE);
        b[i] = MIN_VALUE + (double)rand() / RAND_MAX * (MAX_VALUE - MIN_VALUE);
    }
}

void print_configuration(int N, int chunk_size, int threadsPerBlock) {
    printf("Processing array of size N = %d\n", N);
    if (N >= DEFAULT_CHUNK_SIZE) {
        printf("Using chunk_size = %d (for CUDA), threadsPerBlock = %d\n", 
               chunk_size, threadsPerBlock);
    } else {
        printf("CUDA chunking not applicable for N < default chunk_size. "
               "Using N as chunk_size.\n");
    }
}

int parse_arguments(int argc, char *argv[]) {
    int N = DEFAULT_ARRAY_SIZE;
    int opt;
    
    while ((opt = getopt(argc, argv, "n:")) != -1) {
        switch (opt) {
            case 'n':
                N = atoi(optarg);
                if (N <= 0) {
                    fprintf(stderr, "Error: Array size (N) must be a positive integer.\n");
                    exit(EXIT_FAILURE);
                }
                break;
            default:
                fprintf(stderr, "Usage: %s [-n array_size]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }
    return N;
}

int main(int argc, char *argv[]) {
    // Parse arguments and setup configuration
    const int N = parse_arguments(argc, argv);
    const int threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
    int chunk_size = DEFAULT_CHUNK_SIZE;
    
    if (N < chunk_size) {
        chunk_size = N;
    }
    
    print_configuration(N, chunk_size, threadsPerBlock);

    // Allocate memory for arrays
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
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    // Initialize arrays with random values
    initialize_arrays(a, b, N);

    // Sequential processing
    clock_t start = clock();
    array_operations_sequential(a, b, sum_seq, diff_seq, prod_seq, quot_seq, N);
    const double sequential_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("Sequential time: %.10f \n", sequential_time);

    // Parallel processing with CUDA
    cudaEvent_t start_event, stop_event;
    CHECK(cudaEventCreate(&start_event));
    CHECK(cudaEventCreate(&stop_event));
    
    CHECK(cudaEventRecord(start_event, 0));
    process_in_chunks_cuda(a, b, sum_par, diff_par, prod_par, quot_par, 
                          N, chunk_size, threadsPerBlock);
    CHECK(cudaEventRecord(stop_event, 0));
    CHECK(cudaEventSynchronize(stop_event));
    
    float cuda_time = 0;
    CHECK(cudaEventElapsedTime(&cuda_time, start_event, stop_event));
    printf("Parallel time : %.10f \n", cuda_time / 1000.0f);
    
    CHECK(cudaEventDestroy(start_event));
    CHECK(cudaEventDestroy(stop_event));

    // Cleanup
    free(a); free(b);
    free(sum_seq); free(diff_seq); free(prod_seq); free(quot_seq);
    free(sum_par); free(diff_par); free(prod_par); free(quot_par);

    printf("Processing finished.\n");
    return 0;
}
