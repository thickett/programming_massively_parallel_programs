#include "imageShared.h"
#include <iostream>
#include <cstdlib>

pixelRGB createRandomPixel() {
	pixelRGB p;
	p.r = static_cast<float>(rand() % 256);
	p.g = static_cast<float>(rand() % 256);
	p.b = static_cast<float>(rand() % 256);
	return p;
}
image createRandomImage(int h, int w) {
	std::srand(static_cast<unsigned int>(std::time(nullptr)));
	image img;
	img.height = h;
	img.width = w;


	for (int i = 0; i < h; i++) {
		std::vector<pixelRGB> newRow(img.width);
		for (int j = 0; j < w; j++) {
			newRow[j] = createRandomPixel();
		}
		img.pixelGrid.push_back(newRow);
	}
	return img;
}

void printImage(const image& img) {
	std::cout << "____RGB Image____" << std::endl;
	for (int i = 0; i < img.height; i++) {
		for (int j = 0; j < img.width; j++) {
			const pixelRGB p = img.pixelGrid[i][j];
			std::cout << "( " << p.r << ", " << p.g << ", " << p.b << " )";
		}
		std::cout << std::endl;
	}
}

void printImageGrey(const imageGreyScale& img) {
	std::cout << "____Grey Image____" << std::endl;
	for (int i = 0; i < img.height; i++) {
		for (int j = 0; j < img.width; j++) {
			const pixelGrey& p = img.pixelGrid[i][j];
			std::cout << "( " << p.hue << " )";
		}
		std::cout << std::endl;
	}
}