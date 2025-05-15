#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void quicksort(int *arr, int left, int right) {
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
    quicksort(arr, left, j);
    quicksort(arr, i, right);
}

int main() {
    int n = 1000000;
    int *arr = malloc(n * sizeof(int));
    srand(time(NULL));

    for (int i = 0; i < n; i++) arr[i] = rand();

    clock_t start = clock();
    quicksort(arr, 0, n - 1);
    clock_t end = clock();

    printf("Sequential sort time: %.3f sec\n", (double)(end - start) / CLOCKS_PER_SEC);

    free(arr);
    return 0;
}
