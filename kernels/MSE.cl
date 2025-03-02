// MSE 

__kernel void MSE(
	__global const float *Y,
	__global const float *Y_hat
	__global const float *L,
	const int batch_size,
	const int in_features
){
	int idx = get_global_id(0);
	int N = batch_size * in_features;
	
	if(idx > N) return;

	float pred = Y[idx];
	float truth = Y_hat[idx];
	float err = (pred - truth) * (pred - truth) * 1/N;
	atomic_add(L, err);
}
