#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define ARRAY_SIZE 100000
#define ROOT 0

void elementwise_operations_seq(float *a, float *b, float *sum, float *diff, float *prod, float *div, int size) {
    for (int i = 0; i < size; ++i) {
        sum[i]  = a[i] + b[i];
        diff[i] = a[i] - b[i];
        prod[i] = a[i] * b[i];
        div[i]  = (b[i] != 0.0f) ? (a[i] / b[i]) : 0.0f;
    }
}

void compare_results(const char *label, float *seq, float *par, int size) {
    for (int i = 0; i < size; ++i) {
        if (fabs(seq[i] - par[i]) > 1e-5) {
            printf("❌ Ошибка в %s на индексе %d: %.5f != %.5f\n", label, i, seq[i], par[i]);
            return;
        }
    }
    printf("✅ Результат %s совпадает.\n", label);
}

int main(int argc, char **argv) {
    int rank, size;
    float *a = NULL, *b = NULL;
    float *sum_seq = NULL, *diff_seq = NULL, *prod_seq = NULL, *div_seq = NULL;
    float *sum_par = NULL, *diff_par = NULL, *prod_par = NULL, *div_par = NULL;

    float *a_local, *b_local;
    float *sum_local, *diff_local, *prod_local, *div_local;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (ARRAY_SIZE % size != 0) {
        if (rank == ROOT)
            printf("Ошибка: ARRAY_SIZE должно делиться на количество процессов.\n");
        MPI_Finalize();
        return 1;
    }

    int chunk_size = ARRAY_SIZE / size;

    // Аллоцирование и инициализация массивов
    if (rank == ROOT) {
        a = malloc(sizeof(float) * ARRAY_SIZE);
        b = malloc(sizeof(float) * ARRAY_SIZE);
        sum_seq = malloc(sizeof(float) * ARRAY_SIZE);
        diff_seq = malloc(sizeof(float) * ARRAY_SIZE);
        prod_seq = malloc(sizeof(float) * ARRAY_SIZE);
        div_seq = malloc(sizeof(float) * ARRAY_SIZE);
        sum_par = malloc(sizeof(float) * ARRAY_SIZE);
        diff_par = malloc(sizeof(float) * ARRAY_SIZE);
        prod_par = malloc(sizeof(float) * ARRAY_SIZE);
        div_par = malloc(sizeof(float) * ARRAY_SIZE);

        srand(time(NULL));
        for (int i = 0; i < ARRAY_SIZE; ++i) {
            a[i] = (float)(rand() % 1000);
            b[i] = (float)((rand() % 999) + 1); // избегаем деления на 0
        }

        double start_seq = MPI_Wtime();
        elementwise_operations_seq(a, b, sum_seq, diff_seq, prod_seq, div_seq, ARRAY_SIZE);
        double end_seq = MPI_Wtime();
        printf("⏱️ Последовательная обработка завершена за %.4f секунд.\n", end_seq - start_seq);
    }

    // Выделение памяти для локальных массивов
    a_local = malloc(sizeof(float) * chunk_size);
    b_local = malloc(sizeof(float) * chunk_size);
    sum_local = malloc(sizeof(float) * chunk_size);
    diff_local = malloc(sizeof(float) * chunk_size);
    prod_local = malloc(sizeof(float) * chunk_size);
    div_local = malloc(sizeof(float) * chunk_size);

    // Распределение данных
    MPI_Scatter(a, chunk_size, MPI_FLOAT, a_local, chunk_size, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
    MPI_Scatter(b, chunk_size, MPI_FLOAT, b_local, chunk_size, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

    double start_par = MPI_Wtime();

    for (int i = 0; i < chunk_size; ++i) {
        sum_local[i]  = a_local[i] + b_local[i];
        diff_local[i] = a_local[i] - b_local[i];
        prod_local[i] = a_local[i] * b_local[i];
        div_local[i]  = (b_local[i] != 0.0f) ? (a_local[i] / b_local[i]) : 0.0f;
    }

    // Сбор результатов
    MPI_Gather(sum_local,  chunk_size, MPI_FLOAT, sum_par,  chunk_size, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
    MPI_Gather(diff_local, chunk_size, MPI_FLOAT, diff_par, chunk_size, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
    MPI_Gather(prod_local, chunk_size, MPI_FLOAT, prod_par, chunk_size, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
    MPI_Gather(div_local,  chunk_size, MPI_FLOAT, div_par,  chunk_size, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

    double end_par = MPI_Wtime();

    if (rank == ROOT) {
        printf("⏱️ Параллельная обработка завершена за %.4f секунд.\n", end_par - start_par);

        // Сравнение результатов
        compare_results("сложения", sum_seq, sum_par, ARRAY_SIZE);
        compare_results("вычитания", diff_seq, diff_par, ARRAY_SIZE);
        compare_results("умножения", prod_seq, prod_par, ARRAY_SIZE);
        compare_results("деления", div_seq, div_par, ARRAY_SIZE);

        // Освобождение памяти
        free(a); free(b);
        free(sum_seq); free(diff_seq); free(prod_seq); free(div_seq);
        free(sum_par); free(diff_par); free(prod_par); free(div_par);
    }

    // Освобождение локальных
    free(a_local); free(b_local);
    free(sum_local); free(diff_local); free(prod_local); free(div_local);

    MPI_Finalize();
    return 0;
}
