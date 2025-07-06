#include <iostream>
#include <math.h>

void vecAdd(float *a, float *b, float *c, int n) {
	for (int i = 0; i < n; i++) {
		c[i] = a[i] + b[i];
	}
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
	vecAdd(a, b, c, n);

	float maxError = 0.0f;
	for (int i = 0; i < n; i ++ ) {
		maxError = fmax(maxError, fabs(c[i] - 3.0f));
	}
	std::cout << "max error: " << maxError << std::endl;

	delete[] a;
	delete[] b;
	delete[] c;

	return 0;
}