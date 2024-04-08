#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

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

    // Set up OpenCL environment
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
    program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, (const size_t *)&kernelSize, &err);
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

    // Matrix dimensions
    const int M = 3; // Number of rows in A and C
    const int N = 3; // Number of columns in B and C
    const int K = 3; // Number of columns in A and rows in B

    // Initialize matrices A and B
    float A[M * K];
    float B[K * N];
    float C[M * N]; // Result matrix

    // Populate matrices A and B
    for (int i = 0; i < M * K; ++i) {
        A[i] = i + 1; // Some example data
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = i + 1; // Some example data
    }

    // Create buffer objects for matrices A, B, and C
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * M * K, NULL, &err);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * K * N, NULL, &err);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * M * N, NULL, &err);
    if (!bufferA || !bufferB || !bufferC || err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create buffer objects.\n");
        return EXIT_FAILURE;
    }

    // Write matrices A and B to the device
    err = clEnqueueWriteBuffer(commandQueue, bufferA, CL_TRUE, 0, sizeof(float) * M * K, A, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commandQueue, bufferB, CL_TRUE, 0, sizeof(float) * K * N, B, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to write data to device.\n");
        return EXIT_FAILURE;
    }

    // Set kernel arguments
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

    // Define global and local work size
    size_t globalWorkSize[2] = {M, N};
    size_t localWorkSize[2] = {1, 1};

    // Enqueue kernel for execution
    err = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to enqueue kernel.\n");
        return EXIT_FAILURE;
    }

    // Read result matrix C from the device
    err = clEnqueueReadBuffer(commandQueue, bufferC, CL_TRUE, 0, sizeof(float) * M * N, C, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to read data from device.\n");
        return EXIT_FAILURE;
    }

    // Print result matrix C
    printf("Result Matrix C:\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%f\t", C[i * N + j]);
        }
        printf("\n");
    }

    // Clean up
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
    free(kernelSource);

    return EXIT_SUCCESS;
}
