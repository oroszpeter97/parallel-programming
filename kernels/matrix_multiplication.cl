__kernel void matrixMultiply(__global const float* A,
                              __global const float* B,
                              __global float* C,
                              const int M,
                              const int N,
                              const int K) {
    // Get global thread ID
    int row = get_global_id(0);
    int col = get_global_id(1);

    // Initialize result element
    float result = 0.0f;

    // Compute dot product for each element in the result matrix
    for (int k = 0; k < K; ++k) {
        result += A[row * K + k] * B[k * N + col];
    }

    // Write result to output matrix
    C[row * N + col] = result;
}
