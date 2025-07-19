
#include <iostream>

__global__
void vectorMatrixMultiplyBase(float* matrix_c, float* vector_b, float* vector_a, int N) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < N) {
		float sum = 0.0f;
		for (int i = 0; i < N; i++) {
			sum += matrix_c[N * i + col] * vector_b[i];
		}
		vector_a[col] = sum;

	}
	
}
void printVector(float* vector,int vector_width) {
	for (int i = 0; i < vector_width; i++) {
		std::cout << vector[i] << ", " ;
	}
	std::cout << std::endl;
}
void printMatrix(float* matrix, int rows, int width) {
	std::cout << "output_vector with dimensions: ( " << rows << ", " << width << " )" << std::endl;
	for (int row_num = 0; row_num < rows; row_num++) {
		for (int col_num = 0; col_num < width; col_num++) {
			std::cout << matrix[row_num * width + col_num] << ", ";
		}
		std::cout << std::endl;
	}

}

int main() {
	int N = 4;

	int vector_elements = N;
	int matrix_elements = N * N;

	float* matrix_c;
	float* vector_b;
	float* vector_a;


	cudaMallocManaged(&matrix_c, matrix_elements * sizeof(float));
	cudaMallocManaged(&vector_b, vector_elements *sizeof(float));
	cudaMallocManaged(&vector_a, vector_elements  * sizeof(float));

	for (int i = 0; i < matrix_elements; i++) {
		matrix_c[i] = (float)i+1.0f;
	}
	for (int i = 0; i < vector_elements; i++) {
		vector_b[i] = (float)(i *2 + 1);
	}
	printVector(vector_b, N);
	printMatrix(matrix_c, N, N);
	
	
	vectorMatrixMultiplyBase << <(N+256 -1) / 256, 256 >> > (matrix_c, vector_b, vector_a, N);
	cudaDeviceSynchronize();
	printVector(vector_a, N);

	cudaFree(matrix_c);
	cudaFree(vector_b);
	cudaFree(vector_a);

	return 0;
}