#include <iostream>
#include <random>
#include <cstdlib>
#include "imageShared.h"


static imageGreyScale makeRGBGreyScale(const image& img) {
	imageGreyScale imgGrey;
	imgGrey.height = img.height;
	imgGrey.width = img.width;
	
	for (int i = 0; i < imgGrey.height; i++) {
		std::vector<pixelGrey> newGreyRow(imgGrey.width);
		for (int j = 0; j < imgGrey.width; j++) {
			
			const pixelRGB& pRGB = img.pixelGrid[i][j];
			
			pixelGrey pGrey;
			pGrey.hue = 0.21f * pRGB.r + 0.72f * pRGB.g + pRGB.b * 0.07f;
			newGreyRow[j] = pGrey;
			
		}
		imgGrey.pixelGrid.push_back(newGreyRow);
	}
	return imgGrey;
}



int main(){

int h = 16;
int w = 19;
image randomImage = createRandomImage(h, w);
imageGreyScale randomImageGrey = makeRGBGreyScale(randomImage);

std::cout << "image height: " << randomImage.height << std::endl;
std::cout << "image width: " << randomImage.width << std::endl;
printImage(randomImage);
printImageGrey(randomImageGrey);
return 0;
}