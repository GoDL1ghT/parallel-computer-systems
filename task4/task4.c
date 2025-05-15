#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define ROWS 400
#define COLS 400

void initialize_arrays(double A[ROWS][COLS], double B[ROWS][COLS]) {
    for (int i = 0; i < ROWS; ++i)
        for (int j = 0; j < COLS; ++j) {
            A[i][j] = i + j + 1;
            B[i][j] = (i + 1) * (j + 1);
        }
}

void sequential_operations(
    double A[ROWS][COLS], double B[ROWS][COLS],
    double C_add[ROWS][COLS], double C_sub[ROWS][COLS],
    double C_mul[ROWS][COLS], double C_div[ROWS][COLS]
) {
    for (int i = 0; i < ROWS; ++i)
        for (int j = 0; j < COLS; ++j) {
            C_add[i][j] = A[i][j] + B[i][j];
            C_sub[i][j] = A[i][j] - B[i][j];
            C_mul[i][j] = A[i][j] * B[i][j];
            C_div[i][j] = B[i][j] != 0 ? A[i][j] / B[i][j] : 0;
        }
}

void parallel_operations(
    double A[ROWS][COLS], double B[ROWS][COLS],
    double C_add[ROWS][COLS], double C_sub[ROWS][COLS],
    double C_mul[ROWS][COLS], double C_div[ROWS][COLS]
) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ROWS; ++i)
        for (int j = 0; j < COLS; ++j) {
            C_add[i][j] = A[i][j] + B[i][j];
            C_sub[i][j] = A[i][j] - B[i][j];
            C_mul[i][j] = A[i][j] * B[i][j];
            C_div[i][j] = B[i][j] != 0 ? A[i][j] / B[i][j] : 0;
        }
}

int main() {
    double A[ROWS][COLS], B[ROWS][COLS];
    double C_add[ROWS][COLS], C_sub[ROWS][COLS];
    double C_mul[ROWS][COLS], C_div[ROWS][COLS];

    initialize_arrays(A, B);

    double start, end;

    // Последовательно
    start = omp_get_wtime();
    sequential_operations(A, B, C_add, C_sub, C_mul, C_div);
    end = omp_get_wtime();
    printf("Sequential execution time: %.6f seconds\n", end - start);

    // Параллельно
    start = omp_get_wtime();
    parallel_operations(A, B, C_add, C_sub, C_mul, C_div);
    end = omp_get_wtime();
    printf("Parallel execution time: %.6f seconds\n", end - start);

    return 0;
}
