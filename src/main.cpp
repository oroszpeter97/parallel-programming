#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

std::string readKernelFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open kernel file: " << filename << std::endl;
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main() {
    std::string kernelFilename = "kernels/sum.cl";
    std::string kernelCode = readKernelFromFile(kernelFilename);

    if (kernelCode.empty()) {
        std::cerr << "Kernel code is empty. Exiting..." << std::endl;
        return 1;
    }

    std::cout << "Kernel code:\n" << kernelCode << std::endl;

    return 0;
}
