#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void quicksort(int *arr, int left, int right, int depth) {
    if (left >= right) return;

    int pivot = arr[(left + right) / 2];
    int i = left, j = right;
    while (i <= j) {
        while (arr[i] < pivot) i++;
        while (arr[j] > pivot) j--;
        if (i <= j) {
            int tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
            i++; j--;
        }
    }

    if (depth < 4) {  // Ограничиваем глубину, чтобы не создать слишком много задач
        #pragma omp task
        quicksort(arr, left, j, depth + 1);
        #pragma omp task
        quicksort(arr, i, right, depth + 1);
    } else {
        quicksort(arr, left, j, depth + 1);
        quicksort(arr, i, right, depth + 1);
    }
}

int main() {
    int n = 1000000;
    int *arr = malloc(n * sizeof(int));
    srand(time(NULL));

    for (int i = 0; i < n; i++) arr[i] = rand();

    double start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        quicksort(arr, 0, n - 1, 0);
    }
    double end = omp_get_wtime();

    printf("Parallel sort time: %.3f sec\n", end - start);

    free(arr);
    return 0;
}
