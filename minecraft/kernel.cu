#define SDL_MAIN_HANDLED
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <SDL2/SDL_image.h>
#include <iostream>
#include <chrono>
#include "renderer.cuh"
#include "physics.cuh"

using namespace std;

__device__ uint32_t* d_framebuffer;

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

	sphere(origin,0.5f,h_scene,h_sceneSize,material(diffuse),white_texture);
	
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
	
	float fps = 0;

	TTF_Font* font = nullptr;
	SDL_Texture* fpsTexture = nullptr;
	SDL_Rect fpsRect = {10, 10, 0, 0};  // Position at top-left with some padding
	SDL_Color textColor = {255, 255, 255, 255};  // White text
	if(TTF_Init() == -1) {
		cout << "TTF_Init error: " << TTF_GetError() << endl;
	}

	font = TTF_OpenFont("C:\\Windows\\Fonts\\consola.ttf",24);  // Change path/font size as needed
	if(!font) {
		cout << "Failed to load font: " << TTF_GetError() << endl;
	}

	SDL_Rect widget_rect = {w/2-(180* wid_Size)/2,h-(18* wid_Size )-h/16,180* wid_Size,18* wid_Size};
	SDL_Surface* widget_surf = IMG_Load("..\\textures\\inventory.png");
	if(!widget_surf) {
		printf("Failed to load image: %s\n",IMG_GetError());
		return 1;
	}

	SDL_Texture* widget = SDL_CreateTextureFromSurface(renderer,widget_surf);
	SDL_FreeSurface(widget_surf);  // Free surface after creating texture

	if(!widget) {
		printf("Failed to create texture: %s\n",SDL_GetError());
		return 1;
	}

	while(1) {
		nframe++;
		auto currentTime = chrono::high_resolution_clock::now();
		chrono::duration<float> deltaTime = currentTime - lastTime;
		float dt = deltaTime.count();
		lastTime = currentTime;
		sum_time += deltaTime.count() * 1000;
		if(nframe % 10 == 0) {
			fps = 1000 / (sum_time / 10);
			sum_time = 0;

			stringstream ss;
			ss.precision(1);
			ss << fixed << "FPS: " << fps << " , scene size: " << h_sceneSize << "/" << MAX_OBJ;

			// Free old texture if exists
			if(fpsTexture) {
				SDL_DestroyTexture(fpsTexture);
				fpsTexture = nullptr;
			}

			SDL_Surface* textSurface = TTF_RenderText_Blended(font,ss.str().c_str(),textColor);
			if(textSurface) {
				fpsTexture = SDL_CreateTextureFromSurface(renderer,textSurface);
				SDL_FreeSurface(textSurface);

				if(fpsTexture) {
					SDL_QueryTexture(fpsTexture,nullptr,nullptr,&fpsRect.w,&fpsRect.h);
					// fpsRect.x and .y already set to 10,10
				}
			}
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
				yaw -= e.motion.xrel * mouse_sens;
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
						int faceIdx = cubes[idxCube].sIdx + face * 2;
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
				h_scene[0].a += rotation(0,0,yaw) * move.norm() * curr_move_speed * dt;
			}
			else {
				h_lights[current_light_index] = h_lights[current_light_index] + move.norm() * move_speed * dt;
				cudaMemcpyToSymbol(lights,h_lights,h_lightsSize * sizeof(vec3),0,cudaMemcpyHostToDevice);
			}
		}

		origin = h_scene[0].a;

		handleCollisions(h_scene,h_sceneSize);

		SoA_h_scene->sceneSize = 0;
		for(int i = 0; i < h_sceneSize; i++) { // convert host scene (AoS) to device scene (SoA)
			SoA_h_scene->addObject(h_scene[i]);
		}

		cudaMemcpy(scene,SoA_h_scene,sizeof(Scene),cudaMemcpyHostToDevice);

		int _pitch;
		SDL_LockTexture(texture,nullptr,(void**)&framebuffer,&_pitch);

		dim3 block(8,8);
		dim3 grid((w + block.x - 1) / block.x,(h + block.y - 1) / block.y);


		render_pixel << <grid,block >> > (scene,d_framebuffer,origin,rot,foc_len,move_light,current_light_index);

		cudaError_t err = cudaGetLastError();
		if(err != cudaSuccess) {
			printf("Kernel error: %s\n",cudaGetErrorString(err));
		}


		cudaMemcpy(framebuffer,d_framebuffer,sizeof(uint32_t) * w * h,cudaMemcpyDeviceToHost);

		SDL_UnlockTexture(texture);
		SDL_RenderClear(renderer);
		SDL_RenderCopy(renderer,texture,nullptr,nullptr);
		if(fpsTexture) {
			SDL_RenderCopy(renderer,fpsTexture,nullptr,&fpsRect);
		}
		if(widget) {
			SDL_RenderCopy(renderer,widget,nullptr,&widget_rect);
		}

		SDL_RenderPresent(renderer);
	}
}