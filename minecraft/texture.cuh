#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include "algebra.cuh"
#include <sstream>

#define MAX_TEX_SIZE 256// 64x64

class texture {
private:
	int width,height;
	vec3 matrix[MAX_TEX_SIZE];
	bool _texture;
	vec3 color;
public:
	texture() {};
	texture(bool Texture,vec3 Color = {0,0,0}):_texture(Texture),color(Color) {};
	__host__ void fromFile(const char* filename,const int& Width,const int& Height) {
		width = Width; height = Height;
		FILE* file = fopen(filename,"rb");
		
		if(!file) {
			perror("Error opening file");
		}
		if(Width * Height > MAX_TEX_SIZE) {
			perror("FILE EXCEED MAX TEXTURE SIZE");
		}

		unsigned char R,G,B;
		int i = 0;
		// read three characters at a time
		while(fread(&R,1,1,file) == 1 && fread(&G,1,1,file) == 1 && fread(&B,1,1,file) == 1) {
			matrix[i] = vec3{float(R),float(G),float(B)};
			i++;
		}
		if(i != width * height) perror("NOT ENOUGH PIXELS IN TEXTURE\n");
		fclose(file);

	}
	__device__ vec3 at(float x,float y) const {
		if(!_texture) {
			return color;
		}
		if(x >= 1) x = 0.99;
		if(y >= 1) y = 0.99;
		int idx = (floor(y * height)) * width + floor(x * width);
		return matrix[idx];
	}
	__device__ vec3 at_raw(unsigned int x,unsigned int y) const {
		if(!_texture) {
			return color;
		}
		if(x >= width) x = width-1;
		if(y >= height) y = height-1;
		int idx = y * width + x;
		return matrix[idx];
	}
};

#define IMPORT_TEXTURE(name,dev_name,filename) texture name(true);name.fromFile(filename,16,16);texture* dev_name;cudaMalloc(&dev_name,sizeof(texture));cudaMemcpy(dev_name,&name,sizeof(texture),cudaMemcpyHostToDevice);;
