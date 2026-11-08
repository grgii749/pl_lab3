#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "LU.h"

static const char *KERNEL_SRC =
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    "__kernel void extract_column(__global const double *U,\n"
    "                             const int n, const int k,\n"
    "                             __global double *col) {\n"
    "    int gid = get_global_id(0);\n"
    "    int i = gid + k;\n"
    "    if (i < n) col[gid] = U[i*n + k];\n"
    "}\n"
    "__kernel void swap_rows(__global double *U, __global double *L,\n"
    "                        const int n, const int k, const int pivot) {\n"
    "    int gid = get_global_id(0);\n"
    "    if (gid < n) {\n"
    "        double t = U[k*n + gid];\n"
    "        U[k*n + gid] = U[pivot*n + gid];\n"
    "        U[pivot*n + gid] = t;\n"
    "    }\n"
    "    if (gid < k) {\n"
    "        double t = L[k*n + gid];\n"
    "        L[k*n + gid] = L[pivot*n + gid];\n"
    "        L[pivot*n + gid] = t;\n"
    "    }\n"
    "}\n"
    "__kernel void compute_L(__global const double *U, __global double *L,\n"
    "                        const int n, const int k) {\n"
    "    int i = get_global_id(0) + k + 1;\n"
    "    if (i < n) L[i*n + k] = U[i*n + k] / U[k*n + k];\n"
    "}\n"
    "__kernel void update_U(__global double *U, __global const double *L,\n"
    "                       const int n, const int k) {\n"
    "    int j = get_global_id(0) + k + 1;\n"
    "    int h = get_global_id(1) + k;\n"
    "    if (j < n && h < n) U[j*n + h] -= L[j*n + k] * U[k*n + h];\n"
    "}\n";


static cl_context       g_ctx    = NULL;
static cl_command_queue g_queue  = NULL;
static cl_program       g_prog   = NULL;
static cl_kernel        g_k_col  = NULL;
static cl_kernel        g_k_swap = NULL;
static cl_kernel        g_k_L    = NULL;
static cl_kernel        g_k_U    = NULL;
static int              g_init   = 0;
static int              g_err    = 0;

static int ocl_init(void) {
    if (g_init) return g_err;
    g_init = 1;
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;

    if (clGetPlatformIDs(1, &platform, NULL) != CL_SUCCESS) { g_err = 1; return 1; }
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL) != CL_SUCCESS) {
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL) != CL_SUCCESS) {
            fprintf(stderr, "OpenCL: no devices\n");
            g_err = 1; return 1;
        }
    }

    char exts[4096] = {0};
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(exts), exts, NULL);
    if (!strstr(exts, "cl_khr_fp64")) {
        fprintf(stderr, "OpenCL: cl_khr_fp64 not supported\n");
        g_err = 1; return 1;
    }

    g_ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) { g_err = 1; return 1; }

    g_queue = clCreateCommandQueue(g_ctx, device, 0, &err);
    if (err != CL_SUCCESS) { g_err = 1; return 1; }

    g_prog = clCreateProgramWithSource(g_ctx, 1, &KERNEL_SRC, NULL, &err);
    if (err != CL_SUCCESS) { g_err = 1; return 1; }

    err = clBuildProgram(g_prog, 1, &device, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[16384] = {0};
        clGetProgramBuildInfo(g_prog, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "OpenCL build error:\n%s\n", log);
        g_err = 1; return 1;
    }

    g_k_col  = clCreateKernel(g_prog, "extract_column", &err); if (err) { g_err=1; return 1; }
    g_k_swap = clCreateKernel(g_prog, "swap_rows",      &err); if (err) { g_err=1; return 1; }
    g_k_L    = clCreateKernel(g_prog, "compute_L",      &err); if (err) { g_err=1; return 1; }
    g_k_U    = clCreateKernel(g_prog, "update_U",       &err); if (err) { g_err=1; return 1; }
    return 0;
}




void lu(int matrix_size, double **matrix_U, double **matrix_L, int *perm) {
    if (ocl_init()) {
        fprintf(stderr, "OpenCL init failed, using CPU fallback\n");
        return;
    }

    int n = matrix_size;
     for (int i = 0; i < n; i++) perm[i] = i;
    size_t mat_bytes = (size_t)n * n * sizeof(double);
    double *u_flat = malloc(mat_bytes);
    double *l_flat = malloc(mat_bytes);
    double *col_h  = malloc((size_t)n * sizeof(double));
    if (!u_flat || !l_flat || !col_h) {
        free(u_flat); free(l_flat); free(col_h);
        return;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            u_flat[i*n + j] = matrix_U[i][j];
            l_flat[i*n + j] = matrix_L[i][j];
        }
    }

    cl_int err;
    cl_mem u_buf   = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    mat_bytes, u_flat, &err);
    cl_mem l_buf   = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    mat_bytes, l_flat, &err);
    cl_mem col_buf = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE,
                                    (size_t)n * sizeof(double), NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer failed\n");
        if (u_buf)   clReleaseMemObject(u_buf);
        if (l_buf)   clReleaseMemObject(l_buf);
        if (col_buf) clReleaseMemObject(col_buf);
        free(u_flat); free(l_flat); free(col_h);
        return;
    }
    double t_read = 0, t_kL = 0, t_kU = 0, t_kcol = 0;
    for (int k = 0; k < n; k++) {
        int ncol = n - k;


        size_t gsize_col = (size_t)ncol;
        clSetKernelArg(g_k_col, 0, sizeof(cl_mem), &u_buf);
        clSetKernelArg(g_k_col, 1, sizeof(int),    &n);
        clSetKernelArg(g_k_col, 2, sizeof(int),    &k);
        clSetKernelArg(g_k_col, 3, sizeof(cl_mem), &col_buf);
        clEnqueueNDRangeKernel(g_queue, g_k_col, 1, NULL, &gsize_col, NULL, 0, NULL, NULL);
         
        clEnqueueReadBuffer(g_queue, col_buf, CL_TRUE, 0,(size_t)ncol * sizeof(double), col_h, 0, NULL, NULL);



        int pivot = k;
        double maxabs = fabs(col_h[0]);
        for (int i = 1; i < ncol; i++) {
            if (fabs(col_h[i]) > maxabs) { maxabs = fabs(col_h[i]); pivot = k + i; }
        }
        if (maxabs < 1e-12) {
            printf("Singular matrix\n");
            break;
        }

        if (pivot != k) {
            size_t gsize = (size_t)n;
            clSetKernelArg(g_k_swap, 0, sizeof(cl_mem), &u_buf);
            clSetKernelArg(g_k_swap, 1, sizeof(cl_mem), &l_buf);
            clSetKernelArg(g_k_swap, 2, sizeof(int),    &n);
            clSetKernelArg(g_k_swap, 3, sizeof(int),    &k);
            clSetKernelArg(g_k_swap, 4, sizeof(int),    &pivot);
            clEnqueueNDRangeKernel(g_queue, g_k_swap, 1, NULL, &gsize, NULL, 0, NULL, NULL);

            int t = perm[k]; perm[k] = perm[pivot]; perm[pivot] = t;
        }

        if (n - k - 1 > 0) {
            size_t gsize = (size_t)(n - k - 1);
            clSetKernelArg(g_k_L, 0, sizeof(cl_mem), &u_buf);
            clSetKernelArg(g_k_L, 1, sizeof(cl_mem), &l_buf);
            clSetKernelArg(g_k_L, 2, sizeof(int),    &n);
            clSetKernelArg(g_k_L, 3, sizeof(int),    &k);
        clEnqueueNDRangeKernel(g_queue, g_k_L, 1, NULL, &gsize, NULL, 0, NULL, NULL);
        }


        if (n - k - 1 > 0 && n - k > 0) {
            size_t gsize2[2] = { (size_t)(n - k - 1), (size_t)(n - k) };
            clSetKernelArg(g_k_U, 0, sizeof(cl_mem), &u_buf);
            clSetKernelArg(g_k_U, 1, sizeof(cl_mem), &l_buf);
            clSetKernelArg(g_k_U, 2, sizeof(int),    &n);
            clSetKernelArg(g_k_U, 3, sizeof(int),    &k);
            clEnqueueNDRangeKernel(g_queue, g_k_U, 2, NULL, gsize2, NULL, 0, NULL, NULL);
             
        }
    }

    clEnqueueReadBuffer(g_queue, u_buf, CL_TRUE, 0, mat_bytes, u_flat, 0, NULL, NULL);
    clEnqueueReadBuffer(g_queue, l_buf, CL_TRUE, 0, mat_bytes, l_flat, 0, NULL, NULL);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix_U[i][j] = u_flat[i*n + j];
            matrix_L[i][j] = l_flat[i*n + j];
        }
    }

    clReleaseMemObject(u_buf);
    clReleaseMemObject(l_buf);
    clReleaseMemObject(col_buf);
    free(u_flat); free(l_flat); free(col_h);
}


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
        y[i]=sum/matrix_L[i][i];
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