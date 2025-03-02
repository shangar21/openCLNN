// MSE 

__kernel void MSE(
	__global const float *Y,
	__global const float *Y_hat,
	__global float *L,
	const int batch_size,
	const int in_features
){
	int batch = get_global_id(0); // sample
	int feature = get_global_id(1); // features
	
	if (batch >= batch_size || feature >= in_features) return;
	
	int idx = batch * in_features + feature;
	
	float sq_err = (Y[idx] - Y_hat[idx]) * (Y[idx] - Y_hat[idx]);
	//atomic_add(&L[batch], sq_err);
	L[batch] += sq_err;
	
}
