#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define SIZE 1000000

int main(int argc, char *argv[]) {
    int *array = NULL;
    long long seq_sum = 0;
    double start_time, end_time;

    array = malloc(SIZE * sizeof(int));
    srand(time(NULL));
    for (int i = 0; i < SIZE; i++) {
        array[i] = rand() % 100;
    }

    start_time = MPI_Wtime();
    for (int i = 0; i < SIZE; i++) {
        seq_sum += array[i];
    }
    end_time = MPI_Wtime();

    printf("Sequential total sum: %lld\n", seq_sum);
    printf("Sequential time: %.6f seconds\n", end_time - start_time);

    int rank, size;
    int *sub_array = NULL;
    long long local_sum = 0, total_sum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = SIZE / size;
    sub_array = malloc(chunk_size * sizeof(int));

    // Замер времени только на корневом процессе
    if (rank == 0)
        start_time = MPI_Wtime();

    MPI_Scatter(array, chunk_size, MPI_INT, sub_array, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < chunk_size; i++) {
        local_sum += sub_array[i];
    }

    MPI_Reduce(&local_sum, &total_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("Parallel total sum: %lld\n", total_sum);
        printf("Parallel time: %.6f seconds\n", end_time - start_time);
    }

    free(array);
    free(sub_array);
    MPI_Finalize();
    return 0;
}
