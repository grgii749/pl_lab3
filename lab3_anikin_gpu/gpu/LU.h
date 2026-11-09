#include <stdio.h>
float** get_A(int matrix_size);
float** get_U(int matrix_size);
float** get_L(int matrix_size);
void parse_start_matrix(int matrix_size, float** start_matrix, float** matrix_U, FILE* f);
float* get_b(int matrix_size, FILE* f);
void lu(int matrix_size, float** matrix_U, float** matrix_L, int* perm);
float* solve_sle(int matrix_size, float** matrix_U, float** matrix_L, float* b, int* perm);
float** dif2_to_matrix(int n, float* matrix);
float* dif2(int m, int n);