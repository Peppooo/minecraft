#pragma once
#include "objects.cuh"

#define MAX_OBJ 1000000

struct Scene {
private:
	vec3 t_normal[MAX_OBJ];
public:
	vec3 _min[MAX_OBJ];
	vec3 _max[MAX_OBJ];
	texture* tex[MAX_OBJ];
	normal* norm[MAX_OBJ];
	material mat[MAX_OBJ];
	size_t sceneSize;
	void addObject(const cube& obj) {
		_min[sceneSize] = obj._min;
		_max[sceneSize] = obj._max;
		tex[sceneSize] = obj.tex;
		mat[sceneSize] = obj.mat;
		norm[sceneSize] = obj.norm;
		sceneSize++;
	}
	__device__ __forceinline__ vec3 color(const int idx,const vec3& p) const {
		return tex[idx]->at(p,t_normal[idx]);
	};
	__device__ __forceinline__ bool intersect(const int idx,const vec3& O,const vec3& D,vec3& p,vec3& N,double* dist = nullptr) const
	{
		vec3 invDir = 1 / D;
		vec3 tMin = (_min[idx] - O) * invDir;
		vec3 tMax = (_max[idx] - O) * invDir;
		vec3 t1 = v_min(tMin,tMax);
		vec3 t2 = v_max(tMin,tMax);
		float dstFar = min(min(t2.x,t2.y),t2.z);
		float dstNear = max(max(t1.x,t1.y),t1.z);
		if(dstFar >= dstNear && dstFar > 0) {
			if(dist) {
				*dist = dstNear;
			}

			p = O + D * dstNear;

			vec3 dir_sph = p - ((_min[idx] + _max[idx]) * 0.5f);

			int comp_idx = max_idx(abs(dir_sph));

			vec3 new_N = {0,0,0};

			if(dir_sph[comp_idx]<0)
			{
				new_N[comp_idx] = -1;
			}
			else new_N[comp_idx] = 1;

			p = p + (new_N * 1e-5);
			N = new_N;
			return true;
		};
		return false;
	};
};