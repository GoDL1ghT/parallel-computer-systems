#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void kernel(void) {
	printf("My first program on the Kepler-supercomputer CUDA!\n");
}

int main(void) {
	kernel <<< 3, 3 >>> ();
	return 0;
}