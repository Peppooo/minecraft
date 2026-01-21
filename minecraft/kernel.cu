#define SDL_MAIN_HANDLED
#include <iostream>
#include <chrono>
#include "engine.cuh"
#include "settings.cuh"

using namespace std;

int main() {
	// scene infos
	cube* h_scene = new cube[MAX_OBJ]; size_t h_sceneSize = 0; // scene size calculated step by step 
	light h_lights[1] = {light{{0,1,0},{0.5,0.5,0.5}}}; int h_lightsSize = 1;

	COLOR_TEXTURE(white_texture,(vec3{1,1,1}));
	COLOR_TEXTURE(blue_texture,(vec3{0,0,1}));

	IMPORT_TEXTURE(dirt,"..\\textures\\dirt.tex",vec2(0,0),vec2(1,1),16,16);

	DEFAULT_NORMAL_MAP(default_norm_map);
	
	renderer Camera(1024,1024,M_PI / 1.5f,1,1,1);

	Camera.init("Minecraft");
	Camera.origin = vec3{0,0,0};
	Camera.max_reflections = 3;

	cube({0,0,0},{1,1,1},h_scene,h_sceneSize,material(diffuse),dirt,default_norm_map);
	cube({1,1,0},{2,2,1},h_scene,h_sceneSize,material(diffuse),dirt,default_norm_map);

	int numKeys;
	const Uint8* keystates=SDL_GetKeyboardState(&numKeys);

	SDL_Event e;
	SDL_SetRelativeMouseMode(SDL_TRUE);

	auto lastTime = chrono::high_resolution_clock::now();

	float sum_time = 0;

	Camera.import_scene_from_host_array(h_scene,h_sceneSize,32);
	Camera.import_lights_from_host(h_lights,h_lightsSize);
	
	while(1) {
		if(Camera.frame_n % 3 == 0) {
			cout << "frame time: " << sum_time / 3 << " ms" << endl; // average frame time out of 10
			sum_time = 0;
		}

		while(SDL_PollEvent(&e)) {
			if(e.type == SDL_KEYDOWN) {
				if(e.key.keysym.scancode == SDL_SCANCODE_L) {
					move_light = !move_light;
					if(move_light) {
						current_light_index++;
						current_light_index = cycle(current_light_index,h_lightsSize);
					}
				}
				if(e.key.keysym.scancode == SDL_SCANCODE_H) {
					Camera.ssaa = 4;
				}
			}
			if(e.type == SDL_QUIT) {
				return 0;
			}
			if(e.type == SDL_MOUSEMOTION) {
				Camera.yaw -= e.motion.xrel * mouse_sens;
				Camera.pitch -= e.motion.yrel * mouse_sens;
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
				Camera.origin += rotation(0,0,Camera.yaw) * move.norm() * curr_move_speed * Camera.frame_dt;
			}
			else {
				h_lights[current_light_index].pos = h_lights[current_light_index].pos + move.norm() * curr_move_speed * Camera.frame_dt;
				Camera.import_lights_from_host(h_lights,h_lightsSize);
			}
		}

		Camera.render();

		sum_time += Camera.frame_dt*1000;
	}
}