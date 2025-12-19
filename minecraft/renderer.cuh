#pragma once
#include <stdio.h>
#include "objects.cuh"
#include "settings.cuh"
#include "scene.cuh"

__constant__ vec3 lights[16];

__constant__ int lightsSize;

__host__ __device__ __forceinline__ int castRay(const Scene* scene,const vec3& O,const vec3& D,vec3& p,vec3& n,float& dist) {
	int closestIdx = -1;
	dist = INFINITY;
	float c_dist = 0;
	for(int i = 0; i < scene->sceneSize; i++) {
		//float DIST_ = d_min(d_min((scene->a[i] - O).len2(),(scene->b[i] - O).len2()),(scene->c[i] - O).len2());
		if(dot(scene->t_normal[i],D) > 0) continue;
		if((scene->a[i] - O).len2() > 350) continue;
		vec3 temp_p,temp_n;
		if(scene->intersect(i,O,D,temp_p,temp_n)) {
			c_dist = (temp_p - O).len2();
			if(c_dist < dist) {
				dist = c_dist;
				closestIdx = i;
				p = temp_p;
				n = temp_n;
			}
		}
	}
	return closestIdx;
}

__device__ __forceinline__ vec3 compute_ray(const Scene* scene,vec3 O,vec3 D) {
	vec3 color = {0,0,0};
	vec3 p,surf_norm = {0,0,0}; // p is the intersection location
	float _dist;
	int objIdx = castRay(scene,O,D,p,surf_norm,_dist);
	if(objIdx != -1) {
		color = scene->color(objIdx,p,surf_norm);
	}
	else {
		float kY = (D.norm().y + 1) / 2;
		return vec3{191,245,255}*kY + vec3{0, 110, 255}*(1 - kY);
	}
	return color;
}

__global__ void render_pixel(const Scene* scene,uint32_t* data,vec3 origin,matrix rotation,float focal_length,bool move_light,int current_light_index) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = (x + y * w);
	if(idx >= w * h) return;
	int i = x - w / 2;
	int j = -(y - h / 2);
	vec3 pixel = {0,0,0};

	vec3 dir = rotation * vec3{float(i),float(j),focal_length};
	pixel = compute_ray(scene,origin,dir);
	for(int k = 0; k < lightsSize; k++) {
		float lightdot = dot(dir.norm(),(lights[k] - origin).norm());
		if(lightdot > (1 - 0.0001) && lightdot < (1 + 0.0001)) { // draws a sphere in the location of the light
			pixel = vec3{255,255,255};
			if(current_light_index == k && move_light)
			{
				pixel = vec3{255,0,255};
			}
		}
	}



	data[idx] = pixel.argb();
}