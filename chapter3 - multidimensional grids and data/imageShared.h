#ifndef IMAGE_SHARED_H
#define IMAGE_SHARED_H

#include <vector>

struct pixelRGB
{
	float r;
	float g;
	float b;
};

struct pixelGrey
{
	float hue;
};

struct image
{
	int height;
	int width;
	std::vector<std::vector<pixelRGB>> pixelGrid;
};

struct imageGreyScale
{
	int height;
	int width;
	std::vector<std::vector<pixelGrey>> pixelGrid;
};


// Function declarations
pixelRGB createRandomPixel();
image createRandomImage(int h, int w);
void printImage(const image& img);
void printImageGrey(const imageGreyScale& img);

#endif // IMAGE_SHARED_H