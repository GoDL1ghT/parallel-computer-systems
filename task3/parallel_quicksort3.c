#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000000

int main() {
    float *a = malloc(N * sizeof(float));
    float *b = malloc(N * sizeof(float));
    float *sum = malloc(N * sizeof(float));
    float *diff = malloc(N * sizeof(float));
    float *prod = malloc(N * sizeof(float));
    float *div = malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100 + 1;
        b[i] = rand() % 100 + 1;
    }

    double start = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        sum[i] = a[i] + b[i];
        diff[i] = a[i] - b[i];
        prod[i] = a[i] * b[i];
        div[i] = a[i] / b[i];
    }

    double end = omp_get_wtime();
    printf("Parallel array operations time: %.3f sec\n", end - start);

    free(a); free(b); free(sum); free(diff); free(prod); free(div);
    return 0;
}
