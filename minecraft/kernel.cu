#define SDL_MAIN_HANDLED
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <SDL2/SDL.h>
#include <iostream>
#include <chrono>
#include "renderer.cuh"
#include "physics.cuh"

using namespace std;

__device__ uint32_t* d_framebuffer;

const vec3 gravity = {0,-1,0};

int main() {
	// scene infos
	object* h_scene = new object[MAX_OBJ]; int h_sceneSize = 0; // scene size calculated step by step 
	vec3 h_lights[1] = {{5,4,5}}; int h_lightsSize = 1;

	texture _white_texture(false,{200,200,200});
	texture* white_texture; cudaMalloc(&white_texture,sizeof(texture)); cudaMemcpy(white_texture,&_white_texture,sizeof(texture),cudaMemcpyHostToDevice);

	IMPORT_TEXTURE(stone,d_stone_tex,"..\\textures\\stone.tex");
	IMPORT_TEXTURE(sand,d_sand_tex,"..\\textures\\sand.tex");
	IMPORT_TEXTURE(dirt,d_dirt_tex,"..\\textures\\dirt.tex");

	//int cam_idx = h_sceneSize;
	//sphere({2,2,2},0.5f,h_scene,h_sceneSize,material(diffuse),white_texture);
	
	uint32_t* framebuffer = nullptr;
	
	cudaMalloc(&d_framebuffer,sizeof(uint32_t) * w * h);

	cudaMemcpyToSymbol(lights,h_lights,h_lightsSize * sizeof(vec3),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(lightsSize,&h_lightsSize,sizeof(int),0,cudaMemcpyHostToDevice);
	
	Scene* SoA_h_scene = new Scene();

	SDL_Init(SDL_INIT_EVERYTHING);
	int numKeys;
	const Uint8* keystates=SDL_GetKeyboardState(&numKeys);

	// SDL Initialization
	SDL_Window* window = SDL_CreateWindow("RT",SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,w,h,0);
	SDL_Renderer* renderer = SDL_CreateRenderer(window,-1,SDL_RENDERER_ACCELERATED);
	SDL_Texture* texture = SDL_CreateTexture(
		renderer,
		SDL_PIXELFORMAT_ARGB8888,
		SDL_TEXTUREACCESS_STREAMING,
		w,
		h
	);
	SDL_Event e;
	SDL_SetRelativeMouseMode(SDL_TRUE);

	auto lastTime = chrono::high_resolution_clock::now();
	int nframe = 0;

	float sum_time = 0;
	for(int i = 0; i < 10; i++) {
		newCube({float(i),0,0},h_scene,h_sceneSize,material(diffuse),d_dirt_tex);
	}

	Scene* scene;
	cudaMalloc(&scene,sizeof(Scene));

	SoA_h_scene->sceneSize = 0;
	for(int i = 0; i < h_sceneSize; i++) { // convert host scene (AoS) to device scene (SoA)
		SoA_h_scene->addObject(h_scene[i]);
	}
	cudaMemcpy(scene,SoA_h_scene,sizeof(Scene),cudaMemcpyHostToDevice);
	
	while(1) {
		nframe++;
		auto currentTime = chrono::high_resolution_clock::now();
		chrono::duration<float> deltaTime = currentTime - lastTime;
		float dt = deltaTime.count();
		lastTime = currentTime;
		sum_time += deltaTime.count() * 1000;

		if(nframe % 100 == 0) {
			cout << "FPS " << 1000/(sum_time/100) << "" << endl; // average frame time out of 10
			sum_time = 0;
		}

		auto rot = rotation(0,pitch,yaw);
		while(SDL_PollEvent(&e)) {
			if(e.type == SDL_KEYDOWN) {
				if(keystates[SDL_SCANCODE_L]) {
					move_light = !move_light;
					if(move_light) {
						current_light_index++;
						current_light_index = cycle(current_light_index,h_lightsSize);
					}
				}
				if(keystates[SDL_SCANCODE_H]) {
					hq = !hq;
					cudaMemcpyToSymbol(d_hq,&hq,sizeof(bool),0,cudaMemcpyHostToDevice);
				}
			}
			if(e.type == SDL_QUIT) {
				return 0;
			}
			if(e.type == SDL_MOUSEMOTION) {
				yaw   -= e.motion.xrel * mouse_sens;
				pitch -= e.motion.yrel * mouse_sens;
			}
			if(e.type == SDL_MOUSEBUTTONDOWN) {
				vec3 hitPoint,hitNorm; float hitDist2 = 0;
				int hitObjIdx = castRay(SoA_h_scene,origin,rot.z,hitPoint,hitNorm,hitDist2);
				if(hitDist2 < 6 && hitObjIdx != -1) {
					int face;
					int idxCube = findCube(hitObjIdx,face);
					if(e.button.button == SDL_BUTTON_LEFT) {
						if(idxCube != -1) {
							deleteCube(idxCube,h_scene,h_sceneSize);
						}
					}
					if(e.button.button == 3) {
						int faceIdx = cubes[idxCube].sIdx+face*2;
						newCube(cubes[idxCube].position + h_scene[faceIdx].t_normal,h_scene,h_sceneSize,material(diffuse),d_sand_tex);
					}
				}
			}
		}


		vec3 move = {0,0,0};
		float curr_move_speed = move_speed;
		if(keystates[SDL_SCANCODE_W]) {
			move.z += 1;
		}
		if(keystates[SDL_SCANCODE_S]) {
			move.z -= 1;
		}
		if(keystates[SDL_SCANCODE_D]) {
			move.x += 1;
		}
		if(keystates[SDL_SCANCODE_A]) {
			move.x -= 1;
		}
		if(keystates[SDL_SCANCODE_Q]) {
			move.y -= 1;
		}
		if(keystates[SDL_SCANCODE_E]) {
			move.y += 1;
		}
		if(keystates[SDL_SCANCODE_LSHIFT]) {
			curr_move_speed *= 8;
		}
		if(move.len2() != 0) {
			if(!move_light) {
				origin += rotation(0,0,yaw) * move.norm() * curr_move_speed * dt;
			}
			else {
				h_lights[current_light_index] = h_lights[current_light_index] + move.norm() * move_speed * dt;
				cudaMemcpyToSymbol(lights,h_lights,h_lightsSize * sizeof(vec3),0,cudaMemcpyHostToDevice);
			}
		}

		/*origin = h_scene[cam_idx].a;

		handlePhysics(h_scene,h_sceneSize);

		SoA_h_scene->sceneSize = 0;
		for(int i = 0; i < h_sceneSize; i++) { // convert host scene (AoS) to device scene (SoA)
			SoA_h_scene->addObject(h_scene[i]);
		}

		cudaMemcpy(scene,SoA_h_scene,sizeof(Scene),cudaMemcpyHostToDevice);*/
		
		int _pitch;
		SDL_LockTexture(texture,nullptr,(void**)&framebuffer,&_pitch);
		
		dim3 block(8,8);
		dim3 grid((w + block.x - 1) / block.x,(h + block.y - 1) / block.y);


		SoA_h_scene->sceneSize = 0;
		for(int i = 0; i < h_sceneSize; i++) {
			SoA_h_scene->addObject(h_scene[i]);
		}

		cudaMemcpy(scene,SoA_h_scene,sizeof(Scene),cudaMemcpyHostToDevice);

		render_pixel<<<grid,block>>>(scene,d_framebuffer,origin,rot,foc_len,move_light,current_light_index);

		cudaError_t err = cudaGetLastError();
		if(err != cudaSuccess) {
			printf("Kernel error: %s\n",cudaGetErrorString(err));
		}
		

		cudaMemcpy(framebuffer,d_framebuffer,sizeof(uint32_t)*w*h,cudaMemcpyDeviceToHost);

		SDL_UnlockTexture(texture);
		SDL_RenderClear(renderer);
		SDL_RenderCopy(renderer,texture,nullptr,nullptr);

		SDL_RenderPresent(renderer);
	}
}