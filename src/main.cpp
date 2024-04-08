#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <random>

// TODO: FIX THIS!
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include </opt/rocm/include/CL/cl2.hpp>


void fillMatrixRandom(std::vector<float>& matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = dis(gen);
        }
    }
}

std::string readKernelFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return "";
    }
    std::string kernelSource((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    return kernelSource;
}

int main() {
    // Read kernel source code from file
    std::string kernelSource = readKernelFromFile("kernels/matrix_multiplication.cl");
    if (kernelSource.empty()) {
        return 1;
    }

    // Set up OpenCL environment
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found" << std::endl;
        return 1;
    }

    cl::Platform platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        std::cerr << "No GPU devices found" << std::endl;
        return 1;
    }

    cl::Device device = devices.front();
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    // Build program
    cl::Program::Sources sources;
    sources.push_back({kernelSource.c_str(), kernelSource.length()});
    cl::Program program(context, sources);
    if (program.build({device}) != CL_SUCCESS) {
        std::cerr << "Error building program: "
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return 1;
    }

    // Create kernel
    cl::Kernel kernel(program, "matrixMultiplication");

    // Define matrix dimensions and allocate memory
    const int rows = 10;
    const int cols = 10;
    std::vector<float> A(rows * cols);
    std::vector<float> B(rows * cols);
    std::vector<float> C(rows * cols);

    // Fill matrices A and B with random values
    fillMatrixRandom(A, rows, cols);
    fillMatrixRandom(B, rows, cols);

    // Allocate buffers and transfer data to device
    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * A.size(), A.data());
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * B.size(), B.data());
    cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * C.size());

    // Set kernel arguments
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);
    kernel.setArg(3, rows);
    kernel.setArg(4, cols);

    // Time measurement start
    auto start = std::chrono::high_resolution_clock::now();

    // Execute kernel
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rows, cols), cl::NullRange);

    // Wait for kernel execution to finish
    queue.finish();

    // Time measurement end
    auto end = std::chrono::high_resolution_clock::now();

    // Read result back to host
    queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(float) * C.size(), C.data());

    // Calculate duration
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    // Cleanup
    clReleaseMemObject(bufferA());
    clReleaseMemObject(bufferB());
    clReleaseMemObject(bufferC());

    clReleaseKernel(kernel());
    clReleaseProgram(program());
    clReleaseCommandQueue(queue());
    clReleaseContext(context());

    return 0;
}
