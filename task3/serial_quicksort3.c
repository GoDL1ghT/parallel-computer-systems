#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000000  // 1 миллион элементов

int main() {
    float *a = malloc(N * sizeof(float));
    float *b = malloc(N * sizeof(float));
    float *sum = malloc(N * sizeof(float));
    float *diff = malloc(N * sizeof(float));
    float *prod = malloc(N * sizeof(float));
    float *div = malloc(N * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100 + 1;
        b[i] = rand() % 100 + 1;
    }

    clock_t start = clock();

    for (int i = 0; i < N; i++) {
        sum[i] = a[i] + b[i];
        diff[i] = a[i] - b[i];
        prod[i] = a[i] * b[i];
        div[i] = a[i] / b[i];
    }

    clock_t end = clock();
    printf("Sequential array operations time: %.3f sec\n", (double)(end - start) / CLOCKS_PER_SEC);

    free(a); free(b); free(sum); free(diff); free(prod); free(div);
    return 0;
}
