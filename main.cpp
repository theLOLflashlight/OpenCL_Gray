/*
LodePNG Examples

Copyright (c) 2005-2012 Lode Vandevenne

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software
in a product, an acknowledgment in the product documentation would be
appreciated but is not required.

2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.

3. This notice may not be removed or altered from any source
distribution.
*/

#include "lodepng.h"
#include <iostream>

void convertToGrayscale(std::vector<unsigned char> image, unsigned width, unsigned height);

//Example 1
//Decode from disk to raw pixels with a single function call
void decodeOneStep(const char* filename)
{
	std::vector<unsigned char> image;
	unsigned width, height;

	//decode
	unsigned error = lodepng::decode(image, width, height, filename);

	//if there's an error, display it
	if (error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

	//the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as texture, draw it, ...

	// Where the magic happens
	convertToGrayscale(image, width, height);
}

void encodeOneStep(const char* filename, std::vector<unsigned char>& image, unsigned width, unsigned height)
{
	//Encode the image
	unsigned error = lodepng::encode(filename, image, width, height);

	//if there's an error, display it
	if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
}

void convertToGrayscale(std::vector<unsigned char> image, unsigned width, unsigned height) {
	for (int i = 0; i < width * height * 4; i += 4) {
		// R G B A
		// Luminosity Algo = R * 0.21 + G * 0.72 + B * 0.07
		float gray = (float)image[i] * 0.21 + (float)image[i + 1] * 0.72 + (float)image[i + 2] * 0.07;
		image[i] = gray;
		image[i + 1] = gray;
		image[i + 2] = gray;
		// Alpha stays the same
	}

	encodeOneStep("output.png", image, width, height);
}

int main2(int argc, char *argv[])
{
	const char* filename = argc > 1 ? argv[1] : "input.png";
	decodeOneStep(filename);
    return 0;
}


