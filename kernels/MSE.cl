// MSE 

__kernel void MSE(
	__global const float *Y,
	__global const float *Y_hat,
	__global float *L,
	const int batch_size,
	const int in_features
){
	int batch = get_global_id(0); // batch idx 
	
	if (batch >= batch_size) return;
	
	float sq_err = 0.0;

	for(int idx = 0; idx < in_features; idx++){
		float gt = Y[batch * in_features + idx];
		float out = Y_hat[batch * in_features + idx];
		sq_err += (gt - out) * (gt - out);
	}

	L[batch] = sq_err / (float)in_features;
}
