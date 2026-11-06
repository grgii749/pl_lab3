#include <stdio.h>
#include <stdlib.h>

int ** get_A(int matrix_size){
    int **start_matrix=malloc(matrix_size*sizeof(int*));
    if (!start_matrix){
        return NULL;
    }
    for (int i=0;i<matrix_size;i++){
         start_matrix[i]=malloc(matrix_size*sizeof(int));
         if (!start_matrix[i]){
            free(start_matrix);
        return NULL;
        }
    }
    return start_matrix;
}
int ** get_U(int matrix_size){
    int **matrix_U=malloc(matrix_size*sizeof(int*));
    if (!matrix_U){
        return NULL;
    }
    for (int i=0;i<matrix_size;i++){
         matrix_U[i]=malloc(matrix_size*sizeof(int));
         if (!matrix_U[i]){
            free(matrix_U);
        return NULL;
        }
    }
    return matrix_U;
}


int ** get_L(int matrix_size){
    int **matrix_l=malloc(matrix_size*sizeof(int*));
    if (!matrix_l){
        return NULL;
    }
    for (int i=0;i<matrix_size;i++){
         matrix_l[i]=calloc(matrix_size,sizeof(int));
         if (!matrix_l[i]){
            free(matrix_l);
        return NULL;
        }
    }
    for (int i=0;i<matrix_size;i++){
        for (int j=0;j<matrix_size;j++){
            if (i==j){matrix_l[i][j]=1;}
        }
    }
    return matrix_l;
}
int *get_b(int matrix_size,FILE * f){
    int * rhs=malloc(matrix_size*sizeof(int));
    if (!rhs){
        return NULL;
    }
    for (int i=0;i<matrix_size;i++){
        fscanf(f,"%d",&rhs[i]);
    }
    return rhs;
}
void parse_start_matrix(int matrix_size,int **start_matrix,int **matrix_U,FILE* f){
    int value;
    for (int i=0;i<matrix_size;i++){
        for (int j=0;j<matrix_size;j++){
           fscanf(f,"%d",&value);
           start_matrix[i][j]=value;
           matrix_U[i][j]=value;
           
        }
    }
}


void lu(int matrix_size,int ** matrix_U,int** matrix_L){
    for (int k=0;k<matrix_size;k++){
        for (int i=k+1;i<matrix_size;i++){
            matrix_L[i][k]=matrix_U[i][k]/matrix_U[k][k];
        }
        for (int j=k+1;j<matrix_size;j++){
            for (int h=k;h<matrix_size;h++){
                matrix_U[j][h]=matrix_U[j][h]-matrix_L[j][k]*matrix_U[k][h];
            }
        }
    }
}

// void solve(int matrix_size,int ** matrix_U,int** matrix_L,int *b){

// }

