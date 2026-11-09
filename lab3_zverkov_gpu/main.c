#define CL_TARGET_OPENCL_VERSION 300
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>
#include <CL/cl.h>

const uint32_t IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
};

const uint8_t MSG_TABLE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13}
};

inline static uint32_t rotate_right(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

inline static void g(uint32_t state[16], size_t a, size_t b, size_t c, size_t d, uint32_t mx, uint32_t my) {
    state[a] = state[a] + state[b] + mx;
    state[d] = rotate_right(state[d] ^ state[a], 16);
    state[c] = state[c] + state[d];
    state[b] = rotate_right(state[b] ^ state[c], 12);
    state[a] = state[a] + state[b] + my;
    state[d] = rotate_right(state[d] ^ state[a], 8);
    state[c] = state[c] + state[d];
    state[b] = rotate_right(state[b] ^ state[c], 7);
}

inline static void round_function(uint32_t state[16], uint32_t msg[16], uint32_t r) {
    g(state, 0, 4, 8, 12, msg[MSG_TABLE[r][0]], msg[MSG_TABLE[r][1]]);
    g(state, 1, 5, 9, 13, msg[MSG_TABLE[r][2]], msg[MSG_TABLE[r][3]]);
    g(state, 2, 6, 10, 14, msg[MSG_TABLE[r][4]], msg[MSG_TABLE[r][5]]);
    g(state, 3, 7, 11, 15, msg[MSG_TABLE[r][6]], msg[MSG_TABLE[r][7]]);
    g(state, 0, 5, 10, 15, msg[MSG_TABLE[r][8]], msg[MSG_TABLE[r][9]]);
    g(state, 1, 6, 11, 12, msg[MSG_TABLE[r][10]], msg[MSG_TABLE[r][11]]);
    g(state, 2, 7, 8, 13, msg[MSG_TABLE[r][12]], msg[MSG_TABLE[r][13]]);
    g(state, 3, 4, 9, 14, msg[MSG_TABLE[r][14]], msg[MSG_TABLE[r][15]]);
}

void hash(const uint8_t* input, size_t input_len, uint8_t* output) {
    uint32_t state[16];
    uint32_t block[16] = { 0 };
    memcpy(block, input, input_len);
    for (int i = 0; i < 8; i++) {
        state[i] = IV[i];
    }
    for (int i = 8; i < 12; i++) {
        state[i] = IV[i - 8];
    }
    state[12] = 0;
    state[13] = 0;
    state[14] = (uint32_t)input_len;
    state[15] = 0x08 | 0x01 | 0x02;
    for (int r = 0; r < 7; r++) {
        round_function(state, block, r);
    }
    for (int i = 0; i < 8; i++) {
        uint32_t v = state[i] ^ state[i + 8];
        output[i * 4] = v & 0xFF;
        output[i * 4 + 1] = (v >> 8) & 0xFF;
        output[i * 4 + 2] = (v >> 16) & 0xFF;
        output[i * 4 + 3] = (v >> 24) & 0xFF;
    }
}

int main() {
    int base = 36;
    const char* password = "x9wh7";
    int pass_len = (int)strlen(password);
    uint8_t pass_hash[32];
    hash((uint8_t*)password, pass_len, pass_hash);
    long long combinations = 1;
    for (int i = 0; i < pass_len; i++) {
        combinations *= base;
    }
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    char name[256] = { 0 };
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    printf("OpenCL device: %s\n", name);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
    FILE* f = fopen("kernel.cl", "rb");
    if (!f) {
        printf("File not found");
        return 0;
    }
    fseek(f, 0, SEEK_END);
    long long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* kernel_file = (char*)malloc(file_size + 1);
    fread(kernel_file, 1, file_size, f);
    kernel_file[file_size] = '\0';
    fclose(f);
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_file, NULL, NULL);
    cl_int build_status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (build_status != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("Kernel build error:\n%s\n", log);
        return 0;
    }
    cl_kernel kernel = clCreateKernel(program, "bruteforce", NULL);
    cl_mem buf_pass_hash = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32, pass_hash, NULL);
    cl_mem buf_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 16, NULL, NULL);
    int flag = 0;
    cl_mem buf_found = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &flag, NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_pass_hash);
    clSetKernelArg(kernel, 1, sizeof(int), &pass_len);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_output);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_found);
    clSetKernelArg(kernel, 4, sizeof(long long), &combinations);
    size_t work_dim = (size_t)combinations;
    printf("Starting GPU bruteforce\n");
    QueryPerformanceCounter(&start);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_dim, NULL, 0, NULL, NULL);
    clFinish(queue);
    QueryPerformanceCounter(&end);
    char result[16] = { 0 };
    int found;
    clEnqueueReadBuffer(queue, buf_found, CL_TRUE, 0, sizeof(int), &found, 0, NULL, NULL);
    if (found) {
        clEnqueueReadBuffer(queue, buf_output, CL_TRUE, 0, 16, result, 0, NULL, NULL);
        printf("Password found: %s\n", result);
    }
    else printf("Password not found\n");
    double time = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    printf("Time: %.4f seconds\n", time);
    free(kernel_file);
    clReleaseMemObject(buf_pass_hash);
    clReleaseMemObject(buf_output);
    clReleaseMemObject(buf_found);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}