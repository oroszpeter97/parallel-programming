#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAX_SOURCE_SIZE (0x100000)

void matrixMultiply(const float* A,
                    const float* B,
                    float* C,
                    const int M,
                    const int N,
                    const int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float result = 0.0f;

            for (int k = 0; k < K; ++k) {
                result += A[i * K + k] * B[k * N + j];
            }

            C[i * N + j] = result;
        }
    }
}

double getCurrentTimestamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((double)tv.tv_sec + (double)tv.tv_usec / 1000000.0);
}

int main() {
    // Load the kernel source code from a file
    FILE *kernelFile;
    char *kernelSource;
    size_t kernelSize;

    kernelFile = fopen("kernels/matrix_multiplication.cl", "r");
    if (!kernelFile) {
        fprintf(stderr, "Failed to open kernel file.\n");
        return EXIT_FAILURE;
    }

    kernelSource = (char *)malloc(MAX_SOURCE_SIZE);
    kernelSize = fread(kernelSource, 1, MAX_SOURCE_SIZE, kernelFile);
    fclose(kernelFile);

    printf("Setting up kernel...");
    fflush(stdout);
    cl_platform_id platformId = NULL;
    cl_device_id deviceId = NULL;
    cl_context context;
    cl_command_queue commandQueue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    // Get platform and device
    err = clGetPlatformIDs(1, &platformId, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to get platform ID.\n");
        return EXIT_FAILURE;
    }
    err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &deviceId, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to get device ID.\n");
        return EXIT_FAILURE;
    }

    // Create context
    context = clCreateContext(NULL, 1, &deviceId, NULL, NULL, &err);
    if (!context || err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create context.\n");
        return EXIT_FAILURE;
    }

    // Create command queue
    commandQueue = clCreateCommandQueue(context, deviceId, 0, &err);
    if (!commandQueue || err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create command queue.\n");
        return EXIT_FAILURE;  
    }

    // Create program object
    program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource,
                                        (const size_t *)&kernelSize, &err);
    if (!program || err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create program object.\n");
        return EXIT_FAILURE;
    }

    // Build program
    err = clBuildProgram(program, 1, &deviceId, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to build program.\n");
        return EXIT_FAILURE;
    }

    // Create kernel object
    kernel = clCreateKernel(program, "matrixMultiply", &err);
    if (!kernel || err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create kernel object.\n");
        return EXIT_FAILURE;
    }
    printf("Kernel set up!");
    fflush(stdout);

    printf("Memory allocation...\n");
    fflush(stdout);
    const int M = 1500; // Number of rows in A and C
    const int N = 1500; // Number of columns in B and C
    const int K = 1500; // Number of columns in A and rows in B

    float *A = (float *)malloc(sizeof(float) * M * K);
    float *B = (float *)malloc(sizeof(float) * K * N);
    float *C = (float *)malloc(sizeof(float) * M * N);
    float *D = (float *)malloc(sizeof(float) * M * N);
    printf("Memory allocated!\n");
    fflush(stdout);

    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Failed to allocate memory for matrices.\n");
        return EXIT_FAILURE;
    }

    printf("Populating matrices...\n");
    fflush(stdout);
    for (int i = 0; i < M * K; ++i) {
        A[i] = i + 1; // Some example data
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = i + 1; // Some example data
    }
    printf("Finished populating matrices!\n");
    fflush(stdout);

    printf("Creating buffer objects...\n");
    fflush(stdout);
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                    sizeof(float) * M * K, NULL, &err);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                    sizeof(float) * K * N, NULL, &err);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    sizeof(float) * M * N, NULL, &err);
    if (!bufferA || !bufferB || !bufferC || err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create buffer objects.\n");
        return EXIT_FAILURE;
    }
    printf("Buffer objects created!\n");
    fflush(stdout);

    double startTimeSequential = getCurrentTimestamp();

    printf("Performing regular matrix multiplication...\n");
    fflush(stdout);
    matrixMultiply(A, B, D, M, N, K);
    printf("Regular matrix multiplication finished!\n");
    fflush(stdout);

    double endTimeSequential = getCurrentTimestamp();
    printf("Sequential execution time: %.6f seconds\n", endTimeSequential - startTimeSequential);

    printf("Writing matrices to device...\n");
    fflush(stdout);
    err = clEnqueueWriteBuffer(commandQueue, bufferA, CL_TRUE, 0,
                               sizeof(float) * M * K, A, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commandQueue, bufferB, CL_TRUE, 0,
                                sizeof(float) * K * N, B, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to write data to device.\n");
        return EXIT_FAILURE;
    }
    printf("Matrices written to device!\n");
    fflush(stdout);

    printf("Setting kernel arguments...\n");
    fflush(stdout);
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bufferA);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufferB);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufferC);
    err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&M);
    err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&N);
    err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&K);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to set kernel arguments.\n");
        return EXIT_FAILURE;
    }
    printf("Kernel arguments set!\n");
    fflush(stdout);

    double startTimeOpenCL = getCurrentTimestamp();

    printf("Executing kernel...\n");
    fflush(stdout);
    // Define global and local work size
    size_t globalWorkSize[2] = {M, N};
    size_t localWorkSize[2] = {1, 1};
    err = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize,
                                 localWorkSize, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to enqueue kernel.\n");
        return EXIT_FAILURE;
    }
    printf("Kernel execution finished!\n");
    fflush(stdout);

    double endTimeOpenCL = getCurrentTimestamp();
    printf("OpenCL execution time: %.6f seconds\n", endTimeOpenCL - startTimeOpenCL);

    printf("Reading kernel result buffer...\n");
    fflush(stdout);
    err = clEnqueueReadBuffer(commandQueue, bufferC, CL_TRUE, 0,
                              sizeof(float) * M * N, C, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to read data from device.\n");
        return EXIT_FAILURE;
    }
    printf("Kernel result buffer read!\n");
    fflush(stdout);

    printf("Cleaning up...\n");
    fflush(stdout);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
    free(kernelSource);
    free(A);
    free(B);
    free(C);
    printf("Cleaned up!\n");
    fflush(stdout);

    return EXIT_SUCCESS;
}
