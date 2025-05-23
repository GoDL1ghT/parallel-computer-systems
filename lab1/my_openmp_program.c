#include <stdio.h>
#pragma omp

int main ()
{
  printf ("Befor OpenMP\n");
 
#pragma omp parallel
  {
    printf ("My first program on the Kepler-supercomputer OpenMP\n");
  }
 
  printf ("After OpenMP\n\n");
  return 0;
}
