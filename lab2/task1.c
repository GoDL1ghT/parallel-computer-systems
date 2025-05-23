#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define ARRAY_SIZE 1000000

int main() {
    int* array = malloc(sizeof(int) * ARRAY_SIZE);
    if (!array) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = 1;
    }

    // Последовательно
    double start_seq = omp_get_wtime();
    long long seq_sum = 0;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        seq_sum += array[i];
    }
    double end_seq = omp_get_wtime();

    // Параллельно
    double start_par = omp_get_wtime();
    long long par_sum = 0;
    #pragma omp parallel for reduction(+:par_sum)
    for (int i = 0; i < ARRAY_SIZE; i++) {
        par_sum += array[i];
    }
    double end_par = omp_get_wtime();

    // Результаты
    printf("Sequential time: %lld (time: %.5f seconds)\n", seq_sum, end_seq - start_seq);
    printf("Parallel time: %.5f seconds\n", end_par - start_par);

    free(array);
    return 0;
}
