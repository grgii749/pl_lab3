#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "LU.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        
        return 1;
    }
    FILE* f = fopen(argv[2], "r");
    if (!f) { return 1; }
    fclose(f); 

    for (int matrix_size=100; matrix_size<20000; matrix_size*=2) {
        float* mat = dif2(matrix_size, matrix_size);
        float** matrix_U = dif2_to_matrix(matrix_size, mat); 
        float** matrix_L = get_L(matrix_size);               
        
        float* rhs = malloc(matrix_size * sizeof(float));
        rhs[0] = 1.0f;
        rhs[matrix_size - 1] = 1.0f;
        for (int i = 1; i < matrix_size - 1; i++) {
            rhs[i] = 0.0f;
        }

        int* perm = malloc(matrix_size * sizeof(int));

        clock_t t1 = clock();
        lu(matrix_size, matrix_U, matrix_L, perm);
        clock_t t2 = clock();
        double res = (double)(t2 - t1) / CLOCKS_PER_SEC; 

        printf("n=%d lu time=%g s\n", matrix_size, res);

        float* result = solve_sle(matrix_size, matrix_U, matrix_L, rhs, perm);
        // for (int i=0;i<matrix_size;i++){
        //     printf("%g ",result[i]);
        // }
 
        free(result);
        free(rhs);
        free(perm);
        free(mat); 

        for (int i = 0; i < matrix_size; i++) {
            free(matrix_L[i]);
            free(matrix_U[i]);
        }
        free(matrix_L);
        free(matrix_U);
    }

    return 0;
}