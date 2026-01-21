#include "objects.cuh"
#include <stdio.h>

void handleCollisions(object* scene,const int& sceneSize) {
	for(int i = 0; i < sceneSize; i++) {
		for(int j = i+1; j < sceneSize; j++) {
			object &a = scene[i],&b = scene[j];
			if(a.sphere && b.sphere) {
				float dist = -((b.a - a.a).len()-a.b.x-b.b.x);
				if(dist > 0) {
					vec3 n = (a.a - b.a).norm();
					a.a = a.a + n * (dist / 2 + 0.001f);
					b.a = b.a - n * (dist / 2 + 0.001f);
				}
			}
			else if(a.sphere != b.sphere) {
				float dist = 0; vec3 p = {0,0,0};
				const int &idxTrig = (!scene[j].sphere) ? j : i;
				const int &idxSphere = scene[j].sphere ? j : i;
				trigSphereDist(idxSphere,idxTrig,dist,p,scene);
				if(dist < 0) {
					object &sphere=scene[idxSphere];
					vec3 n = p - sphere.a; // trig == a , sphere == b
					sphere.a=p-n.norm()*(sphere.b.x+0.001f);
				}
			}
		}
	}
}