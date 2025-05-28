#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_SIZE 100000
#define ROOT 0

// Функция для обмена элементов
void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Последовательная пузырьковая сортировка
void bubble_sort_seq(int *array, int size) {
    for (int i = 0; i < size - 1; ++i) {
        for (int j = 0; j < size - i - 1; ++j) {
            if (array[j] > array[j + 1]) {
                swap(&array[j], &array[j + 1]);
            }
        }
    }
}

// Локальная сортировка пузырьком
void bubble_sort_local(int *array, int size) {
    for (int i = 0; i < size - 1; ++i) {
        for (int j = 0; j < size - i - 1; ++j) {
            if (array[j] > array[j + 1]) {
                swap(&array[j], &array[j + 1]);
            }
        }
    }
}

// Функция для вывода массива (используется только для маленьких размеров)
void print_array(const char *msg, int *array, int size) {
    printf("%s:\n", msg);
    for (int i = 0; i < size; ++i) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    int rank, size;
    int *full_array = NULL;
    int *local_array = NULL;
    int chunk_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    chunk_size = ARRAY_SIZE / size;

    // Генерация и сортировка последовательного массива только на ROOT
    if (rank == ROOT) {
        full_array = malloc(ARRAY_SIZE * sizeof(int));
        srand(time(NULL));
        for (int i = 0; i < ARRAY_SIZE; ++i) {
            full_array[i] = rand() % 100000;
        }

        int *copy_array = malloc(ARRAY_SIZE * sizeof(int));
        for (int i = 0; i < ARRAY_SIZE; ++i) {
            copy_array[i] = full_array[i];
        }

        double start_seq = MPI_Wtime();
        bubble_sort_seq(copy_array, ARRAY_SIZE);
        double end_seq = MPI_Wtime();

        printf("Последовательная сортировка завершена за %f секунд.\n", end_seq - start_seq);
        free(copy_array);
    }

    // Выделяем память для локального куска
    local_array = malloc(chunk_size * sizeof(int));

    // Распределение массива по процессам
    MPI_Scatter(full_array, chunk_size, MPI_INT,
                local_array, chunk_size, MPI_INT,
                ROOT, MPI_COMM_WORLD);

    double start_par = MPI_Wtime();

    // Локальная сортировка
    bubble_sort_local(local_array, chunk_size);

    // Параллельная odd-even сортировка
    for (int phase = 0; phase < size; ++phase) {
        if (phase % 2 == 0) {
            if (rank % 2 == 0 && rank + 1 < size) {
                int *neigh = malloc(chunk_size * sizeof(int));
                MPI_Sendrecv(local_array, chunk_size, MPI_INT, rank + 1, 0,
                             neigh, chunk_size, MPI_INT, rank + 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int *merged = malloc(2 * chunk_size * sizeof(int));
                int i = 0, j = 0, k = 0;
                while (i < chunk_size && j < chunk_size) {
                    if (local_array[i] < neigh[j]) merged[k++] = local_array[i++];
                    else merged[k++] = neigh[j++];
                }
                while (i < chunk_size) merged[k++] = local_array[i++];
                while (j < chunk_size) merged[k++] = neigh[j++];
                for (int i = 0; i < chunk_size; ++i)
                    local_array[i] = merged[i];
                free(neigh);
                free(merged);
            } else if (rank % 2 == 1) {
                int *neigh = malloc(chunk_size * sizeof(int));
                MPI_Sendrecv(local_array, chunk_size, MPI_INT, rank - 1, 0,
                             neigh, chunk_size, MPI_INT, rank - 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int *merged = malloc(2 * chunk_size * sizeof(int));
                int i = 0, j = 0, k = 0;
                while (i < chunk_size && j < chunk_size) {
                    if (neigh[i] < local_array[j]) merged[k++] = neigh[i++];
                    else merged[k++] = local_array[j++];
                }
                while (i < chunk_size) merged[k++] = neigh[i++];
                while (j < chunk_size) merged[k++] = local_array[j++];
                for (int i = 0; i < chunk_size; ++i)
                    local_array[i] = merged[chunk_size + i];
                free(neigh);
                free(merged);
            }
        } else {
            if (rank % 2 == 1 && rank + 1 < size) {
                int *neigh = malloc(chunk_size * sizeof(int));
                MPI_Sendrecv(local_array, chunk_size, MPI_INT, rank + 1, 0,
                             neigh, chunk_size, MPI_INT, rank + 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int *merged = malloc(2 * chunk_size * sizeof(int));
                int i = 0, j = 0, k = 0;
                while (i < chunk_size && j < chunk_size) {
                    if (local_array[i] < neigh[j]) merged[k++] = local_array[i++];
                    else merged[k++] = neigh[j++];
                }
                while (i < chunk_size) merged[k++] = local_array[i++];
                while (j < chunk_size) merged[k++] = neigh[j++];
                for (int i = 0; i < chunk_size; ++i)
                    local_array[i] = merged[i];
                free(neigh);
                free(merged);
            } else if (rank % 2 == 0 && rank > 0) {
                int *neigh = malloc(chunk_size * sizeof(int));
                MPI_Sendrecv(local_array, chunk_size, MPI_INT, rank - 1, 0,
                             neigh, chunk_size, MPI_INT, rank - 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int *merged = malloc(2 * chunk_size * sizeof(int));
                int i = 0, j = 0, k = 0;
                while (i < chunk_size && j < chunk_size) {
                    if (neigh[i] < local_array[j]) merged[k++] = neigh[i++];
                    else merged[k++] = local_array[j++];
                }
                while (i < chunk_size) merged[k++] = neigh[i++];
                while (j < chunk_size) merged[k++] = local_array[j++];
                for (int i = 0; i < chunk_size; ++i)
                    local_array[i] = merged[chunk_size + i];
                free(neigh);
                free(merged);
            }
        }
    }

    // Сбор отсортированных кусков обратно
    MPI_Gather(local_array, chunk_size, MPI_INT,
               full_array, chunk_size, MPI_INT,
               ROOT, MPI_COMM_WORLD);

    double end_par = MPI_Wtime();

    if (rank == ROOT) {
        printf("Параллельная сортировка завершена за %f секунд.\n", end_par - start_par);
        // print_array("Отсортированный массив", full_array, 100); // Только если нужно
        free(full_array);
    }

    free(local_array);
    MPI_Finalize();
    return 0;
}
