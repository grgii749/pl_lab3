#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double ** get_A(int matrix_size){
    double **start_matrix=malloc(matrix_size*sizeof(double*));
    if (!start_matrix){
        return NULL;
    }
    for (int i=0;i<matrix_size;i++){
         start_matrix[i]=malloc(matrix_size*sizeof(double));
         if (!start_matrix[i]){
            free(start_matrix);
        return NULL;
        }
    }
    return start_matrix;
}
double ** get_U(int matrix_size){
    double **matrix_U=malloc(matrix_size*sizeof(double*));
    if (!matrix_U){
        return NULL;
    }
    for (int i=0;i<matrix_size;i++){
         matrix_U[i]=malloc(matrix_size*sizeof(double));
         if (!matrix_U[i]){
            free(matrix_U);
        return NULL;
        }
    }
    return matrix_U;
}


double ** get_L(int matrix_size){
    double **matrix_l=malloc(matrix_size*sizeof(double*));
    if (!matrix_l){
        return NULL;
    }
    for (int i=0;i<matrix_size;i++){
         matrix_l[i]=calloc(matrix_size,sizeof(double));
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
double *get_b(int matrix_size,FILE * f){
    double * rhs=malloc(matrix_size*sizeof(double));
    if (!rhs){
        return NULL;
    }
    for (int i=0;i<matrix_size;i++){
        fscanf(f,"%lf",&rhs[i]);
    }
    return rhs;
}
void parse_start_matrix(int matrix_size,double **start_matrix,double **matrix_U,FILE* f){
    double value;
    for (int i=0;i<matrix_size;i++){
        for (int j=0;j<matrix_size;j++){
           fscanf(f,"%lf",&value);
           start_matrix[i][j]=value;
           matrix_U[i][j]=value;
           
        }
    }
}


void lu(int matrix_size,double ** matrix_U,double** matrix_L,int *perm){
    for (int i = 0; i < matrix_size; i++) perm[i] = i;
    for (int k=0;k<matrix_size;k++){
        int pivot = k;
        double maxabs = fabs(matrix_U[k][k]);
        for (int i = k + 1; i < matrix_size; i++) {
            double v = fabs(matrix_U[i][k]);
            if (v > maxabs) {
                maxabs = v;
                pivot = i;
            }
        }
        if (maxabs < 1e-12) {
            printf("Singular matrix\n");
            return;
        }
        if (pivot != k) {
            double *tmp_row = matrix_U[k]; 
            matrix_U[k] = matrix_U[pivot]; 
            matrix_U[pivot] = tmp_row;

            for (int j = 0; j < k; j++) {
                double t = matrix_L[k][j];
                 matrix_L[k][j] = matrix_L[pivot][j]; 
                 matrix_L[pivot][j] = t;
            }

            int t = perm[k]; 
            perm[k] = perm[pivot]; 
            perm[pivot] = t;
        }
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

double * solve_sle(int matrix_size,double ** matrix_U,double** matrix_L,double *b,int *perm){
    double *bp = malloc(matrix_size * sizeof(double));
    if (!bp) return NULL;
    for (int i = 0; i < matrix_size; i++) bp[i] = b[perm[i]]; 


    double *y=calloc(matrix_size,sizeof(double));
    if (!y){return NULL;}
    double *x=calloc(matrix_size,sizeof(double));
    if (!x){return NULL;}
    for (int i=0;i<matrix_size;i++){
        double sum=bp[i];
        for (int j=0;j<i;j++){
            sum=sum-matrix_L[i][j]*y[j];
        }
        y[i]=sum/matrix_L[i][i];;
    }
    for (int i=matrix_size-1;i>=0;i--){
        double sum=y[i];
        for (int j=i+1;j<matrix_size;j++){
            sum=sum-matrix_U[i][j]*x[j];
        }
        x[i]=sum/matrix_U[i][i];
    }
    free(y);
    return x;
}

