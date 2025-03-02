// Sigmoid

__kernel void Sigmoid(
    __global const float *X,
    __global float *Y,
		const int batch_size,
    const int in_features,
    const int out_features
){
    int row = get_global_id(0);
    int col = get_global_id(1);

    if(row >= batch_size || col >= out_features) return;

    int idx = row * out_features + col;

    Y[idx] = 1.0f / (1.0f + exp(-X[idx]));
}

