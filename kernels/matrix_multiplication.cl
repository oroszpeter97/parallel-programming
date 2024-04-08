__kernel void matrixMultiplication(__global float* A,
                                   __global float* B,
                                   __global float* C,
                                   const int rows,
                                   const int cols) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    float sum = 0.0f;
    for (int k = 0; k < cols; ++k) {
        sum += A[row * cols + k] * B[k * cols + col];
    }
    
    C[row * cols + col] = sum;
}
