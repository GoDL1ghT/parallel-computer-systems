#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ROWS 250
#define COLS 400  // 250 * 400 = 100000

void fill_matrix(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)(rand() % 100 + 1);
    }
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int total_elements = ROWS * COLS;
    int chunk_size = total_elements / size;

    double* A = NULL;
    double* B = NULL;

    if (rank == 0) {
        A = (double*)malloc(total_elements * sizeof(double));
        B = (double*)malloc(total_elements * sizeof(double));

        srand(time(NULL));
        fill_matrix(A, ROWS, COLS);
        fill_matrix(B, ROWS, COLS);

        // sequential version
        double* seq_add = (double*)malloc(total_elements * sizeof(double));
        double* seq_sub = (double*)malloc(total_elements * sizeof(double));
        double* seq_mul = (double*)malloc(total_elements * sizeof(double));
        double* seq_div = (double*)malloc(total_elements * sizeof(double));

        double start_seq = MPI_Wtime();
        for (int i = 0; i < total_elements; i++) {
            seq_add[i] = A[i] + B[i];
            seq_sub[i] = A[i] - B[i];
            seq_mul[i] = A[i] * B[i];
            seq_div[i] = A[i] / B[i];
        }
        double end_seq = MPI_Wtime();
        printf("Sequential time: %f seconds\n", end_seq - start_seq);

        free(seq_add);
        free(seq_sub);
        free(seq_mul);
        free(seq_div);
    }

    // parallel version
    double* result_add = (double*)malloc(chunk_size * sizeof(double));
    double* result_sub = (double*)malloc(chunk_size * sizeof(double));
    double* result_mul = (double*)malloc(chunk_size * sizeof(double));
    double* result_div = (double*)malloc(chunk_size * sizeof(double));

    double* local_A = (double*)malloc(chunk_size * sizeof(double));
    double* local_B = (double*)malloc(chunk_size * sizeof(double));

    MPI_Scatter(A, chunk_size, MPI_DOUBLE, local_A, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, chunk_size, MPI_DOUBLE, local_B, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_parallel = MPI_Wtime();

    for (int i = 0; i < chunk_size; i++) {
        result_add[i] = local_A[i] + local_B[i];
        result_sub[i] = local_A[i] - local_B[i];
        result_mul[i] = local_A[i] * local_B[i];
        result_div[i] = local_A[i] / local_B[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_parallel = MPI_Wtime();

    if (rank == 0) {
        double* full_add = (double*)malloc(total_elements * sizeof(double));
        double* full_sub = (double*)malloc(total_elements * sizeof(double));
        double* full_mul = (double*)malloc(total_elements * sizeof(double));
        double* full_div = (double*)malloc(total_elements * sizeof(double));

        MPI_Gather(result_add, chunk_size, MPI_DOUBLE, full_add, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(result_sub, chunk_size, MPI_DOUBLE, full_sub, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(result_mul, chunk_size, MPI_DOUBLE, full_mul, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(result_div, chunk_size, MPI_DOUBLE, full_div, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        printf("Parallel time: %f seconds\n", end_parallel - start_parallel);

        free(full_add);
        free(full_sub);
        free(full_mul);
        free(full_div);
    } else {
        MPI_Gather(result_add, chunk_size, MPI_DOUBLE, NULL, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(result_sub, chunk_size, MPI_DOUBLE, NULL, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(result_mul, chunk_size, MPI_DOUBLE, NULL, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(result_div, chunk_size, MPI_DOUBLE, NULL, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    free(local_A);
    free(local_B);
    free(result_add);
    free(result_sub);
    free(result_mul);
    free(result_div);

    if (rank == 0) {
        free(A);
        free(B);
    }

    MPI_Finalize();
    return 0;
}
