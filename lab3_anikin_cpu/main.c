#include <stdio.h>
#include <stdlib.h>
#include "LU.h"
//#include "test_mat.h"

int main(int argc, char *argv[]){
    if (argc<3){
        printf("No matrix size");
        return 1;
    }
    int matrix_size=atoi(argv[1]);
    FILE *f=fopen(argv[2],"r");
    if (!f){
        return 1;
    }
    //double *mat=dif2(matrix_size,matrix_size);
    double **matrix_A=get_A(matrix_size);
    
    double **matrix_U=get_U(matrix_size);
    
    double **matrix_L=get_L(matrix_size);
    parse_start_matrix(matrix_size,matrix_A,matrix_U,f);

    double *rhs=get_b(matrix_size,f);
    
    int *perm =malloc(matrix_size*sizeof(int));
    lu(matrix_size,matrix_U,matrix_L,perm);
    double *result=solve_sle(matrix_size,matrix_U,matrix_L,rhs,perm);
    for (int i=0;i<matrix_size;i++){
        printf("%g ",result[i]);
    }
    
    
    free(result);
    free(rhs);
    for (int i=0;i<matrix_size;i++){
        free(matrix_L[i]);
    }
    free(matrix_L);

    for (int i=0;i<matrix_size;i++){
        free(matrix_U[i]);
    }
    free(matrix_U);

    for (int i=0;i<matrix_size;i++){
        free(matrix_A[i]);
    }
    free(matrix_A);
    fclose(f);
    return 0;
}