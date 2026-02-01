#pragma once
#include "algebra.cuh"
#include "material.cuh"
#include "texture.cuh"
#include "normal.cuh"

class cube { 
public:
	vec3 _min,_max;
	vec3 d_color;
	material mat;
	texture* tex;
	normal* norm;
	bool isPlayer;
	__host__ cube() {};
	__host__ cube(const vec3& Min,const vec3& Max,cube* scene,size_t& sceneSize,const material& Mat,texture* Tex,normal* Norm):
	_min(Min),_max(Max),mat(Mat),tex(Tex),norm(Norm)
	{
		isPlayer = false;
		scene[sceneSize] = *this;
		sceneSize++;
	}
	vec3 center() const {
		return (_min + _max) * 0.5f;
	}
	vec3& operator[](const int idx) {
		if(idx == 0) return _min;
		return _max;
	}
};