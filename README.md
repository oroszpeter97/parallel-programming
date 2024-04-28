# Assigment
Compare the execution time of big matrix multiplications using CPU and GPU with OpenCL.

## Usage

### Compilation

To compile the program, simply run:

```bash
make all
```

This will compile the source code, create necessary directories, and generate the executable file in the `bin` directory.

### Cleaning

To clean up generated files, run:

```bash
make clean
```

This will remove all object files and the executable.

### Running

To execute the program, run:

```bash
make run
```

This will compile the program (if necessary) and execute it, displaying the sum of all numbers in the array.

## Dependencies

To run this program in a Linux environment, you need:

- A C++ compiler (e.g., g++)
- OpenCL development libraries (usually provided by the vendor, such as the OpenCL SDK from AMD or NVIDIA)
- Make utility

Ensure that the necessary OpenCL headers and libraries are properly installed in your system. If using a package manager like `apt`, you can install OpenCL development packages with:

```bash
sudo apt-get install ocl-icd-opencl-dev
```

Make sure to check the documentation or installation instructions provided by your OpenCL platform provider for specific details on installing and configuring the OpenCL development environment.
