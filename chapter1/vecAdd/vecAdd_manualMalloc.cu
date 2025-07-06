#include <iostream>
#include <math.h>

__global__
void vecAdd(float* a, float* b, float* c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		c[i] = a[i] + b[i];
	}
	//for (int i = 0; i < n; i++) {
	//	c[i] = a[i] + b[i];
	//}
}

int main() {
	int n = 1000;
	float* a = new float[n];
	float* b = new float[n];
	float* c = new float[n];


	for (int i = 0; i < n; i++) {
		a[i] = 1.0f;
		b[i] = 2.0f;
	}
	int array_size = n * sizeof(float);
	float* a_d, * b_d, * c_d;
	cudaMalloc((void**)&a_d, array_size);
	cudaMalloc((void**)&b_d, array_size);
	cudaMalloc((void**)&c_d, array_size);

	cudaMemcpy(a_d, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(c_d, c, n * sizeof(float), cudaMemcpyHostToDevice);

	int gridDim = ceil(n / 128.0);
	int blockDim = 128;

	std::cout << "grid dim: " << gridDim << std::endl;
	std::cout << "block dim: " << blockDim << std::endl;
	std::cout << "grid size: " << gridDim * blockDim << std::endl;

	vecAdd<<<gridDim, blockDim>>>(a_d, b_d, c_d, n);
	cudaDeviceSynchronize();

	cudaMemcpy(c, c_d, n * sizeof(float), cudaMemcpyDeviceToHost);

	float maxError = 0.0f;
	for (int i = 0; i < n; i++) {
		maxError = fmax(maxError, fabs(c[i] - 3.0f));
	}
	std::cout << "c 895 " << c[895]<< std::endl;
	std::cout << "c 896 " << c[896] << std::endl;
	std::cout << "c 897 " << c[897] << std::endl;
	std::cout << "max error: " << maxError << std::endl;
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
	delete[] a;
	delete[] b;
	delete[] c;

	return 0;
}