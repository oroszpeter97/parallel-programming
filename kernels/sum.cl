__kernel void arraySum(__global const float* input, __global float* output, const int arraySize) {
    int gid = get_global_id(0);
    
    if (gid < arraySize) {
        output[gid] = 0.0f;
        for (int i = 0; i <= gid; ++i) {
            output[gid] += input[i];
        }
    }
}
