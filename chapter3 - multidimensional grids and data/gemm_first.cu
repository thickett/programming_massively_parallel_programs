#include <iostream>
#include <random>
__global__
void gemm_single_point(float* matrix_a, float* matrix_b, float* matrix_product, int matrix_width) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if ((col < matrix_width) && (row < matrix_width)) {
			float outputValue = 0;
			for (int k = 0; k < matrix_width;k++) {
				outputValue += matrix_a[row * matrix_width + k] * matrix_b[k * matrix_width + col];
			}
			matrix_product[row * matrix_width + col] = outputValue;
	}
}
__global__
void gemm_single_row(float* matrix_a, float* matrix_b, float* matrix_product, int matrix_width) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < matrix_width) {
		for (int i = 0; i < matrix_width; i++) {
			float sum = 0.0f;
			for (int j = 0; j < matrix_width; j++) {
				sum += matrix_a[matrix_width * row + j] * matrix_b[matrix_width * j + i];
			}
			matrix_product[row * matrix_width + i] = sum;
		}
	}

}

__global__
void gemm_single_col(float* matrix_a, float* matrix_b, float* matrix_product, int matrix_width) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < matrix_width) {
		for (int row = 0; row < matrix_width; row++) {
			float sum = 0.0f;
			for (int k = 0; k < matrix_width; k++) {
				sum += matrix_a[row * matrix_width + k] * matrix_b[k * matrix_width + col];
			}
			matrix_product[row * matrix_width + col] = sum;
		}
	}
}



void printMatrix(float* matrix, int rows, int width) {
	std::cout << "output_vector with dimensions: ( " << rows<< ", " << width <<" )" << std::endl;
	for (int row_num = 0; row_num < rows; row_num++) {
		for (int col_num = 0; col_num < width; col_num++) {
			std::cout << matrix[row_num * width + col_num] <<", ";
		}
		std::cout << std::endl;
	}

}

int main(int argc, char* argv[]) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(0.0f, 10.0f);

	int N = 32;
	if (argc > 1) {
		N = std::atoi(argv[1]);
	}
	int size = N * N * sizeof(float);

	float* h_a = (float*)malloc(size);
	float* h_b = (float*)malloc(size);
	float* h_c = (float*)malloc(size);
	float* h_d = (float*)malloc(size);
	float* h_e = (float*)malloc(size);
	cudaMallocManaged(&h_a, size);
	cudaMallocManaged(&h_b, size);
	cudaMallocManaged(&h_c, size);
	cudaMallocManaged(&h_d, size);
	cudaMallocManaged(&h_e, size);

	for (int i = 0; i < N * N; i++) {
		h_a[i] = dist(gen);
		h_b[i] = dist(gen);
	}


	// single thread single data point implementation
	// no shared data so the ideal setup is anything that utalises 100% of available threads. 
	// ideally we would instansiate a NxM thread grid, where N and M are matrix dimensions
	int blockSize = std::min(N,32);
	dim3 dimBlock(blockSize, blockSize,1); 
	dim3 dimGrid((N + blockSize -1) / blockSize, (N + blockSize - 1) / blockSize);
	gemm_single_point << <dimGrid, dimBlock >> > (h_a, h_b, h_c, N);
	cudaDeviceSynchronize();
	printMatrix(h_c, N, N);

	// single thread per output row
	// ideal dimension is (N) where N is the height of the output matrix
	gemm_single_row << <(N + blockSize - 1) / blockSize, blockSize >> > (h_a, h_b, h_d, N);
	
	cudaDeviceSynchronize();
	printMatrix(h_d, N, N);

	// single thread per output column
	gemm_single_col << <(N + blockSize - 1) / blockSize, blockSize >> > (h_a, h_b, h_e, N);
	cudaDeviceSynchronize();
	printMatrix(h_e, N, N);


	cudaFree(h_a);
	cudaFree(h_b);
	cudaFree(h_c);
	return 0;
}