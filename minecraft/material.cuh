#include "algebra.cuh"

enum material_type {
	diffuse,
	specular,
	glossy
};

__device__ vec3 randomVecInHemisphere(const vec3& n,curandStatePhilox4_32_10_t* state) {
	vec3 v;
	while(true) {
		v = randomVec(state) * 2 - vec3{1,1,1};
		if(dot(v,n) > 0) return v;
	}
}

class material {
public:
	material_type type;
	float roughness;
	float emission;
	material() {};
	material(material_type Type,float Roughness = 0,float Emission = 0): type(Type),emission(Emission),roughness(Roughness) {};
	__device__ bool needs_sampling() const {
		return type == glossy;
	}
	__device__ __forceinline__ vec3 bounce(const vec3& D,vec3 n,curandStatePhilox4_32_10_t* state) const {
		if(type == specular) {
			return D-n*2*dot(D,n);
		}
		else if(type == glossy) {
			// Perfect reflection
			vec3 r = (D - n * 2.0f * dot(D,n)).norm();

			// Convert roughness -> exponent
			float exp = max(0.001f,2.0f / (roughness * roughness) - 2.0f);

			// Importance sample Phong lobe
			float u1 = curand_uniform(state);
			float u2 = curand_uniform(state);

			float cosTheta = powf(u1,1.0f / (exp + 1.0f));
			float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
			float phi = 2.0f * M_PI * u2;

			vec3 t = any_perpendicular(r);
			vec3 b = cross(r,t);

			vec3 wi =
				t * (cosf(phi) * sinTheta) +
				b * (sinf(phi) * sinTheta) +
				r * cosTheta;

			return wi.norm();
		}
		else if(type==diffuse) {
			float r1 = curand_uniform(state);
			float r2 = curand_uniform(state);

			float phi = 2.0f * M_PI * r1;
			float r = sqrt(r2);

			float x = r * cos(phi);
			float y = r * sin(phi);
			float z = sqrt(1.0f - r2);

			vec3 t = any_perpendicular(n);
			vec3 b = cross(t,n);
			
			return (t*x + b*y + n*z);
		}
		return vec3{0,0,0};
	}
	__device__ __forceinline__ vec3 brdf(const vec3& wo,const vec3& n,vec3& wi,const vec3& albedo,float& pdf,bool& delta,curandStatePhilox4_32_10_t* state) const {
		wi = bounce(wo,n,state);
		if(type == specular) {
			delta = true;
			pdf = 1;
			return albedo;
		}
		if(type == diffuse) {
			delta = false;
			pdf = dot(n,wi) / M_PI;
			return albedo/M_PI;
		}

		pdf = 0.0f;
		delta = false;
		return vec3{0,0,0};
	}
};