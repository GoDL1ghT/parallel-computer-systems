#include <stdio.h>
#include "mpi.h"

int main(int argc, char **argv)
{
  printf("Before MPI_INIT\n");
  MPI_Init(&argc, &argv);
  printf("My first program on the Kepler-supercomputer MPI\n");
  MPI_Finalize();
  printf("After MPI_FINALIZE\n");
  return 0;
}
