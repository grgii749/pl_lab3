#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "LU.h"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <n> <matrix_file>\n", argv[0]);
        return 1;
    }
    int matrix_size = atoi(argv[1]);
    FILE *f = fopen(argv[2], "r");
    if (!f) { printf("Cannot open %s\n", argv[2]); return 1; }

    double **matrix_A = get_A(matrix_size);
    double **matrix_U = get_U(matrix_size);
    double **matrix_L = get_L(matrix_size);
    parse_start_matrix(matrix_size, matrix_A, matrix_U, f);

    double *rhs = get_b(matrix_size, f);
    int *perm = malloc(matrix_size * sizeof(int));
    clock_t t1=clock();
    lu(matrix_size, matrix_U, matrix_L, perm);
    clock_t t2=clock();
    double res=(double)(t2-t1)/CLOCKS_PER_SEC;
    printf("lu time%lf",res);
    double *result = solve_sle(matrix_size, matrix_U, matrix_L, rhs, perm);

    //for (int i = 0; i < matrix_size; i++) printf("%g ", result[i]);
    printf("\n");
    
    free(result);
    free(rhs);
    free(perm);
    for (int i = 0; i < matrix_size; i++) { free(matrix_L[i]); free(matrix_U[i]); free(matrix_A[i]); }
    free(matrix_L); free(matrix_U); free(matrix_A);
    fclose(f);
    return 0;
}