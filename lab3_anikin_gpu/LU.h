

#include <stdio.h>

double **get_A(int matrix_size);
double **get_U(int matrix_size);
double **get_L(int matrix_size);

void parse_start_matrix(int matrix_size, double **start_matrix,
                        double **matrix_U, FILE *f);
double *get_b(int matrix_size, FILE *f);

void lu(int matrix_size, double **matrix_U, double **matrix_L, int *perm);

double *solve_sle(int matrix_size, double **matrix_U, double **matrix_L,
                  double *b, int *perm);

