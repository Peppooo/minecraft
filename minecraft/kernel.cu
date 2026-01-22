#define SDL_MAIN_HANDLED
#include <iostream>
#include <chrono>
#include "engine.cuh"
#include "settings.cuh"
#include "game.cuh"

using namespace std;

int main() {
	// scene infos
	cube* h_scene = new cube[MAX_OBJ]; size_t h_sceneSize = 0; // scene size calculated step by step 
	light h_lights[1] = {light{{0,1,0},{.9,.9,.9}}}; int h_lightsSize = 1;

	COLOR_TEXTURE(white_texture,(vec3{1,1,1}));
	COLOR_TEXTURE(blue_texture,(vec3{0,0,1}));

	DEFAULT_NORMAL_MAP(default_norm_map);

	load_textures(default_norm_map);
	
	renderer Camera(1024,1024,M_PI / 1.5f,1,1,1);

	Camera.init("Minecraft");
	Camera.origin = vec3{0,1.5f,0};
	Camera.max_reflections = 4;




	/*for(int i = 0; i < 64; i++) {
		for(int j = 0; j < 64; j++) {
			cube({(float)i,0,(float)j},{i + 1.0f,1,j+1.0f},h_scene,h_sceneSize,material(diffuse),block_textures[i%(types::blocks::count)],default_norm_map);
		}
	}*/

	cube({-1000,-2000,-1000},{1000,0,1000},h_scene,h_sceneSize,material(diffuse),block_textures[blocks::grass],default_norm_map);

	for(float k = 0; k < 3; k++) {

		place_block({0,k,0},blocks::oak_log,h_scene,h_sceneSize);
		place_block({1,k,0},blocks::brick,h_scene,h_sceneSize);

		place_block({3,k,0},blocks::brick,h_scene,h_sceneSize);
		place_block({4,k,0},blocks::oak_log,h_scene,h_sceneSize);

		for(int i = 1; i < 6; i++) {
			for(int j = 0; j <= 5; j += 4) {
				if(k != 1 || (i<=1 || i>=5)) {
					place_block({float(j),k,float(i)},blocks::brick,h_scene,h_sceneSize);
				}
			}
		}

		place_block({0,k,6},blocks::oak_log,h_scene,h_sceneSize);
		place_block({4,k,6},blocks::oak_log,h_scene,h_sceneSize);

		for(float i = 1; i < 4; i++) {
			if(k!=1 || i != 2) place_block({i,k,6},blocks::brick,h_scene,h_sceneSize);
		}

	}

	place_block({2,2,0},blocks::brick,h_scene,h_sceneSize);
	
	cube({0,-7,0},{5,0.001,7},h_scene,h_sceneSize,material(diffuse),block_textures[blocks::cobblestone],default_norm_map);


	int numKeys;
	const Uint8* keystates=SDL_GetKeyboardState(&numKeys);

	SDL_Event e;
	SDL_SetRelativeMouseMode(SDL_TRUE);

	auto lastTime = chrono::high_resolution_clock::now();

	float sum_time = 0;

	Camera.import_scene_from_host_array(h_scene,h_sceneSize,32);
	Camera.import_lights_from_host(h_lights,h_lightsSize);
	
	while(1) {
		if(Camera.frame_n % 5 == 0) {
			cout << "frame time: " << sum_time / 5 << " ms" << endl; // average frame time out of 5
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
			if(e.type == SDL_MOUSEBUTTONDOWN) {
				cout << (int)e.button.button << endl;
				if(e.button.button == SDL_BUTTON_RIGHT) {
					vec3 w_n,_;
					int result_idx = Camera.tree.castRay(Camera.host_soa_scene,Camera.origin,Camera.direction,_,w_n,true);
					if(result_idx != (-1)) {
						place_block(Camera.host_soa_scene->_min[result_idx] + w_n,blocks::cobblestone,h_scene,h_sceneSize);
						Camera.import_scene_from_host_array(h_scene,h_sceneSize,32);
					}
				}
				if(e.button.button == SDL_BUTTON_LEFT) {
					vec3 _;
					int result_idx = Camera.tree.castRay(Camera.host_soa_scene,Camera.origin,Camera.direction,_,_,true);
					if(result_idx != (-1)) {
						swap(h_scene[result_idx],h_scene[h_sceneSize - 1]);
						h_sceneSize--;
						Camera.import_scene_from_host_array(h_scene,h_sceneSize,32);
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