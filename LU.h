#include <stdio.h>
void parse_start_matrix(int matrix_size,int **start_matrix,int **matrix_U,FILE* f);
int ** get_L(int matrix_size);
int ** get_A(int matrix_size);
int ** get_U(int matrix_size);
void lu(int matrix_size,int ** matrix_U,int** matrix_L);
int *get_b(int matrix_size,FILE * f);