#include <stdio.h>
#include <stdlib.h>
#include "LU.h"

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
    
    int **matrix_A=get_A(matrix_size);
    int **matrix_U=get_U(matrix_size);
    int **matrix_L=get_L(matrix_size);
    parse_start_matrix(matrix_size,matrix_A,matrix_U,f);
    int *rhs=get_b(matrix_size,f);
    for (int i=0;i<matrix_size;i++){
        printf("%d ",rhs[i]);
    }
    lu(matrix_size,matrix_U,matrix_L);
    
    
    
    for (int i=0;i<matrix_size;i++){
        for (int j=0;j<matrix_size;j++){
            printf("%d ",matrix_U[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    for (int i=0;i<matrix_size;i++){
        for (int j=0;j<matrix_size;j++){
            printf("%d ",matrix_L[i][j]);
        }
        printf("\n");
    }

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