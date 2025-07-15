#include <iostream>
#include <math.h>

__global__
void vecAdd(float* a, float* b, float* c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		c[i] = a[i] + b[i];
	}
}

int main() {
	int n = 1000;
	float* a, * b, * c;
	int array_size = n * sizeof(float);
	cudaMallocManaged(&a, array_size);
	cudaMallocManaged(&b, array_size);
	cudaMallocManaged(&c, array_size);

	for (int i = 0; i < n; i++) {
		a[i] = 1.0f;
		b[i] = 2.0f;
	}
	
	int gridDim = ceil(n / 128.0);
	int blockDim = 128;

	std::cout << "grid dim: " << gridDim << std::endl;
	std::cout << "block dim: " << blockDim << std::endl;
	std::cout << "grid size: " << gridDim * blockDim << std::endl;

	vecAdd << <gridDim, blockDim >> > (a, b, c, n);
	cudaDeviceSynchronize();

	float maxError = 0.0f;
	for (int i = 0; i < n; i++) {
		maxError = fmax(maxError, fabs(c[i] - 3.0f));
	}
	std::cout << "c 895 " << c[895] << std::endl;
	std::cout << "c 896 " << c[896] << std::endl;
	std::cout << "c 897 " << c[897] << std::endl;
	std::cout << "max error: " << maxError << std::endl;
	
	
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	return 0;
}