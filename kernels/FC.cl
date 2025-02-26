// Y = XW^T 

__kernel void FC(
	__global const float *X,
	__global const float *W,
	__global float *Y,
	const int batch_size,
	const int in_features,
	const int out_features
){
	int row = get_global_id(0);
	int col = get_global_id(1);
	
	if (row >= batch_size || col >= out_features) return;
	
	float sum = 0.0f;
	
	for (int k = 0; k < in_features; k++){
		sum += W[col * in_features + k] * X[row * in_features + k];
	}

	Y[row * out_features + col] = sum;
}
