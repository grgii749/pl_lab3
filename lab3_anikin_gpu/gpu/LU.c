#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "LU.h"

static const char* KERNEL_SRC =
"__kernel void extract_column(__global const float *U,\n"
"                             const int n, const int k,\n"
"                             __global float *col) {\n"
"    int gid = get_global_id(0);\n"
"    int i = gid + k;\n"
"    if (i < n) col[gid] = U[i*n + k];\n"
"}\n"
"__kernel void swap_rows(__global float *U, __global float *L,\n"
"                        const int n, const int k, const int pivot) {\n"
"    int gid = get_global_id(0);\n"
"    if (gid < n) {\n"
"        float t = U[k*n + gid];\n"
"        U[k*n + gid] = U[pivot*n + gid];\n"
"        U[pivot*n + gid] = t;\n"
"    }\n"
"    if (gid < k) {\n"
"        float t = L[k*n + gid];\n"
"        L[k*n + gid] = L[pivot*n + gid];\n"
"        L[pivot*n + gid] = t;\n"
"    }\n"
"}\n"
"__kernel void compute_L(__global const float *U, __global float *L,\n"
"                        const int n, const int k) {\n"
"    int i = get_global_id(0) + k + 1;\n"
"    if (i < n) L[i*n + k] = U[i*n + k] / U[k*n + k];\n"
"}\n"
"__kernel void update_U(__global float *U, __global const float *L,\n"
"                       const int n, const int k) {\n"
"    int j = get_global_id(0) + k + 1;\n"
"    int h = get_global_id(1) + k;\n"
"    if (j < n && h < n) U[j*n + h] -= L[j*n + k] * U[k*n + h];\n"
"}\n";

static cl_context       g_ctx = NULL;
static cl_command_queue g_queue = NULL;
static cl_program       g_prog = NULL;
static cl_kernel        g_k_col = NULL;
static cl_kernel        g_k_swap = NULL;
static cl_kernel        g_k_L = NULL;
static cl_kernel        g_k_U = NULL;
static int              g_init = 0;
static int              g_err = 0;

static int ocl_init(void) {
    if (g_init) return g_err;
    g_init = 1;
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;

    if (clGetPlatformIDs(1, &platform, NULL) != CL_SUCCESS) { g_err = 1; 
        return 1; 
    }
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL) != CL_SUCCESS) {
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL) != CL_SUCCESS) {
            printf( "no device\n");
            g_err = 1; 
            return 1;
        }
    }

    g_ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) { g_err = 1; 
        return 1;
    }

    g_queue = clCreateCommandQueue(g_ctx, device, 0, &err);
    if (err != CL_SUCCESS) { g_err = 1; 
        return 1;     
    }

    g_prog = clCreateProgramWithSource(g_ctx, 1, &KERNEL_SRC, NULL, &err);
    if (err != CL_SUCCESS) { g_err = 1; 
        return 1; 
    }


    err = clBuildProgram(g_prog, 1, &device, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        g_err = 1; 
        return 1;
    }

    g_k_col = clCreateKernel(g_prog, "extract_column", &err); 
    if (err) { 
        g_err = 1; 
        return 1; 
    }
    g_k_swap = clCreateKernel(g_prog, "swap_rows", &err); 
    if (err) { 
        g_err = 1;
        return 1; 
    }
    g_k_L = clCreateKernel(g_prog, "compute_L", &err);
     if (err) 
    { g_err = 1; 
        return 1; 
    }
    g_k_U = clCreateKernel(g_prog, "update_U", &err); 
    if (err) { 
        g_err = 1; 
        return 1; 
    }
    return 0;
}

void lu(int matrix_size, float** matrix_U, float** matrix_L, int* perm) {
    if (ocl_init()) {
        printf("OpenCL init failed\n");
        return;
    }

    int n = matrix_size;
    for (int i = 0; i < n; i++) {
        perm[i] = i;
    }
    size_t mat_bytes = (size_t)n * n * sizeof(float);

    float* u_flat = malloc(mat_bytes);
    float* l_flat = malloc(mat_bytes);
    float* col_h = malloc((size_t)n * sizeof(float));
    if (!u_flat || !l_flat || !col_h) {
        free(u_flat); 
        free(l_flat); 
        free(col_h);
        return;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            u_flat[i * n + j] = matrix_U[i][j];
            l_flat[i * n + j] = matrix_L[i][j];
        }
    }

    cl_int err;
    cl_mem u_buf = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mat_bytes, u_flat, &err);
    cl_mem l_buf = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mat_bytes, l_flat, &err);
    cl_mem col_buf = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, (size_t)n * sizeof(float), NULL, &err);

    if (err != CL_SUCCESS) {
        printf("clCreateBuffer failed\n");
        if (u_buf)   {
            clReleaseMemObject(u_buf);
        }
        if (l_buf)   {
            clReleaseMemObject(l_buf);
        }
        if (col_buf) {
            clReleaseMemObject(col_buf);
        }
        free(u_flat); 
        free(l_flat); 
        free(col_h);
        return;
    }

    for (int k = 0; k < n; k++) {
        int ncol = n - k;

        size_t gsize_col = (size_t)ncol;
        clSetKernelArg(g_k_col, 0, sizeof(cl_mem), &u_buf);
        clSetKernelArg(g_k_col, 1, sizeof(int), &n);
        clSetKernelArg(g_k_col, 2, sizeof(int), &k);
        clSetKernelArg(g_k_col, 3, sizeof(cl_mem), &col_buf);
        clEnqueueNDRangeKernel(g_queue, g_k_col, 1, NULL, &gsize_col, NULL, 0, NULL, NULL);

        clEnqueueReadBuffer(g_queue, col_buf, CL_TRUE, 0, (size_t)ncol * sizeof(float), col_h, 0, NULL, NULL);

        int pivot = k;
        float maxabs = fabsf(col_h[0]);
        for (int i = 1; i < ncol; i++) {
            if (fabsf(col_h[i]) > maxabs) { 
                maxabs = fabsf(col_h[i]); 
                pivot = k + i; }
        }

        if (maxabs < 1e-6f) {
            printf("Singular matrix\n");
            break;
        }

        if (pivot != k) {
            size_t gsize = (size_t)n;
            clSetKernelArg(g_k_swap, 0, sizeof(cl_mem), &u_buf);
            clSetKernelArg(g_k_swap, 1, sizeof(cl_mem), &l_buf);
            clSetKernelArg(g_k_swap, 2, sizeof(int), &n);
            clSetKernelArg(g_k_swap, 3, sizeof(int), &k);
            clSetKernelArg(g_k_swap, 4, sizeof(int), &pivot);
            clEnqueueNDRangeKernel(g_queue, g_k_swap, 1, NULL, &gsize, NULL, 0, NULL, NULL);

            int t = perm[k]; 
            perm[k] = perm[pivot]; 
            perm[pivot] = t;
        }

        if (n - k - 1 > 0) {
            size_t gsize = (size_t)(n - k - 1);
            clSetKernelArg(g_k_L, 0, sizeof(cl_mem), &u_buf);
            clSetKernelArg(g_k_L, 1, sizeof(cl_mem), &l_buf);
            clSetKernelArg(g_k_L, 2, sizeof(int), &n);
            clSetKernelArg(g_k_L, 3, sizeof(int), &k);
            clEnqueueNDRangeKernel(g_queue, g_k_L, 1, NULL, &gsize, NULL, 0, NULL, NULL);
        }

        if (n - k - 1 > 0) {
            size_t gsize2[2] = { (size_t)(n - k - 1), (size_t)(n - k) };
            clSetKernelArg(g_k_U, 0, sizeof(cl_mem), &u_buf);
            clSetKernelArg(g_k_U, 1, sizeof(cl_mem), &l_buf);
            clSetKernelArg(g_k_U, 2, sizeof(int), &n);
            clSetKernelArg(g_k_U, 3, sizeof(int), &k);
            clEnqueueNDRangeKernel(g_queue, g_k_U, 2, NULL, gsize2, NULL, 0, NULL, NULL);
        }
    }

    clEnqueueReadBuffer(g_queue, u_buf, CL_TRUE, 0, mat_bytes, u_flat, 0, NULL, NULL);
    clEnqueueReadBuffer(g_queue, l_buf, CL_TRUE, 0, mat_bytes, l_flat, 0, NULL, NULL);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix_U[i][j] = u_flat[i * n + j];
            matrix_L[i][j] = l_flat[i * n + j];
        }
    }

    clReleaseMemObject(u_buf);
    clReleaseMemObject(l_buf);
    clReleaseMemObject(col_buf);
    free(u_flat); 
    free(l_flat); 
    free(col_h);
}

float** get_A(int matrix_size) {
    float** start_matrix = malloc(matrix_size * sizeof(float*));
    if (!start_matrix) {
        return NULL;
    }
    for (int i = 0; i < matrix_size; i++) {
        start_matrix[i] = malloc(matrix_size * sizeof(float));
        if (!start_matrix[i]) {
            for (int k = 0; k < i; k++) {
                free(start_matrix[k]);
            }
            free(start_matrix);
            return NULL;
        }
    }
    return start_matrix;
}

float** get_U(int matrix_size) {
    return get_A(matrix_size);
}

float** get_L(int matrix_size) {
    float** matrix_l = malloc(matrix_size * sizeof(float*));
    if (!matrix_l) {
        return NULL;
    }

    for (int i = 0; i < matrix_size; i++) {
        matrix_l[i] = calloc(matrix_size, sizeof(float));
        if (!matrix_l[i]) {
            for (int k = 0; k < i; k++) {
                free(matrix_l[k]);
            }

            free(matrix_l);
            return NULL;
        }
    }
    for (int i = 0; i < matrix_size; i++) {
        matrix_l[i][i] = 1.0f;
    }
    return matrix_l;
}

float* get_b(int matrix_size, FILE* f) {
    float* rhs = malloc(matrix_size * sizeof(float));
    if (!rhs) {
        return NULL;
    }

    for (int i = 0; i < matrix_size; i++) {
        fscanf(f, "%f", &rhs[i]); 
    }
    return rhs;
}

void parse_start_matrix(int matrix_size, float** start_matrix, float** matrix_U, FILE* f) {
    float value;
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            fscanf(f, "%f", &value); 
            start_matrix[i][j] = value;
            matrix_U[i][j] = value;
        }
    }
}

float* solve_sle(int matrix_size, float** matrix_U, float** matrix_L, float* b, int* perm) {
    float* bp = malloc(matrix_size * sizeof(float));
    if (!bp) {
        return NULL;
    }
    for (int i = 0; i < matrix_size; i++) {
        bp[i] = b[perm[i]];
    }

    float* y = calloc(matrix_size, sizeof(float));
    if (!y) { 
        free(bp);
         return NULL; 
        }

    float* x = calloc(matrix_size, sizeof(float));
    if (!x) { 
        free(bp);
         free(y);
          return NULL;
    }

    for (int i = 0; i < matrix_size; i++) {
        float sum = bp[i];
        for (int j = 0; j < i; j++) {
            sum = sum - matrix_L[i][j] * y[j];
        }
        y[i] = sum /matrix_L[i][i];
    }

    for (int i = matrix_size - 1; i >= 0; i--) {
        float sum = y[i];
        for (int j = i + 1; j < matrix_size; j++) {
            sum = sum - matrix_U[i][j] * x[j];
        }
        x[i] = sum / matrix_U[i][i];
    }

    free(bp);
    free(y);
    return x;
}

float* dif2(int m, int n)

/******************************************************************************/
/*
  Purpose:

    DIF2 returns the DIF2 matrix.

  Example:

    N = 5

    2 -1  .  .  .
   -1  2 -1  .  .
    . -1  2 -1  .
    .  . -1  2 -1
    .  .  . -1  2

  Properties:

    A is banded, with bandwidth 3.
    A is tridiagonal.
    Because A is tridiagonal, it has property A (bipartite).
    A is a special case of the TRIS or tridiagonal scalar matrix.
    A is integral, therefore det ( A ) is integral, and
    det ( A ) * inverse ( A ) is integral.
    A is Toeplitz: constant along diagonals.
    A is symmetric: A' = A.
    Because A is symmetric, it is normal.
    Because A is normal, it is diagonalizable.
    A is persymmetric: A(I,J) = A(N+1-J,N+1-I).
    A is positive definite.
    A is an M matrix.
    A is weakly diagonally dominant, but not strictly diagonally dominant.
    A has an LU factorization A = L * U, without pivoting.
      The matrix L is lower bidiagonal with subdiagonal elements:
        L(I+1,I) = -I/(I+1)
      The matrix U is upper bidiagonal, with diagonal elements
        U(I,I) = (I+1)/I
      and superdiagonal elements which are all -1.
    A has a Cholesky factorization A = L * L', with L lower bidiagonal.
      L(I,I) =    sqrt ( (I+1) / I )
      L(I,I-1) = -sqrt ( (I-1) / I )
    The eigenvalues are
      LAMBDA(I) = 2 + 2 * COS(I*PI/(N+1))
                = 4 SIN^2(I*PI/(2*N+2))
    The corresponding eigenvector X(I) has entries
       X(I)(J) = sqrt(2/(N+1)) * sin ( I*J*PI/(N+1) ).
    Simple linear systems:
      x = (1,1,1,...,1,1),   A*x=(1,0,0,...,0,1)
      x = (1,2,3,...,n-1,n), A*x=(0,0,0,...,0,n+1)
    det ( A ) = N + 1.
    The value of the determinant can be seen by induction,
    and expanding the determinant across the first row:
      det ( A(N) ) = 2 * det ( A(N-1) ) - (-1) * (-1) * det ( A(N-2) )
                = 2 * N - (N-1)
                = N + 1

    The family of matrices is nested as a function of N.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    30 September 2010

  Author:

    John Burkardt

  Reference:

    Robert Gregory, David Karney,
    A Collection of Matrices for Testing Computational Algorithms,
    Wiley, 1969,
    ISBN: 0882756494,
    LC: QA263.68

    Morris Newman, John Todd,
    Example A8,
    The evaluation of matrix inversion programs,
    Journal of the Society for Industrial and Applied Mathematics,
    Volume 6, Number 4, pages 466-476, 1958.

    John Todd,
    Basic Numerical Mathematics,
    Volume 2: Numerical Algebra,
    Birkhauser, 1980,
    ISBN: 0817608117,
    LC: QA297.T58.

    Joan Westlake,
    A Handbook of Numerical Matrix Inversion and Solution of
    Linear Equations,
    John Wiley, 1968,
    ISBN13: 978-0471936756,
    LC: QA263.W47.

  Parameters:

    Input, int M, N, the order of the matrix.

    Output, float DIF2[M*N], the matrix.
*/
{
    float* a;
    int i;
    int j;

    a = (float*)malloc(m * n * sizeof(float));

    for (j = 0; j < n; j++)
    {
        for (i = 0; i < m; i++)
        {
            if (j == i - 1)
            {
                a[i + j * m] = -1.0;
            }
            else if (j == i)
            {
                a[i + j * m] = 2.0;
            }
            else if (j == i + 1)
            {
                a[i + j * m] = -1.0;
            }
            else
            {
                a[i + j * m] = 0.0;
            }
        }
    }

    return a;
}









float** dif2_to_matrix(int n, float* matrix) {


    float** A = malloc(n * sizeof(float*));
    if (!A) return NULL;

    for (int i = 0; i < n; i++) {
        A[i] = malloc(n * sizeof(float));
        if (!A[i]) {
            for (int k = 0; k < i; k++) free(A[k]);
            free(A);
            return NULL;
        }
        for (int j = 0; j < n; j++) {
            A[i][j] = matrix[i + j * n];
        }
    }
    return A;
}