#define SDL_MAIN_HANDLED
#include <iostream>
#include <chrono>
#include "engine.cuh"
#include "settings.cuh"
#include "game.cuh"
#include "network.h"

using namespace std;

int main() {
	// scene infos
	cube* h_scene = new cube[MAX_OBJ]; size_t h_sceneSize = 0; // scene size calculated step by step 
	light h_lights[1] = {light{{0,1,0},{1,1,1}}}; int h_lightsSize = 0;

	COLOR_TEXTURE(white_texture,(vec3{1,1,1}));
	COLOR_TEXTURE(blue_texture,(vec3{0,0,1}));

	DEFAULT_NORMAL_MAP(default_norm_map);

	
	renderer Camera(512,512,512,512,M_PI / 1.3f,1,1,1);

	printf("initializing camera... ");
	Camera.init("Minecraft"); printf("done\n");

	Camera.origin = vec3{0,3.0f,0};
	Camera.max_reflections = 4;
	Camera.n_samples_pixel = 1;

	printf("loading textures... ");
	load_textures(Camera.sdl_renderer,default_norm_map); printf("done\n");

	inventory _inventory;

	_inventory.selected = 0;

	for(int i = 0; i < 9; i++) {
		_inventory._items[i] = (items)i;
	}
	_inventory._items[8] = glowstone_item;

	int numKeys;
	const Uint8* keystates=SDL_GetKeyboardState(&numKeys);

	SDL_Event e;
	SDL_SetRelativeMouseMode(SDL_TRUE);

	auto lastTime = chrono::high_resolution_clock::now();

	float sum_time = 0;

	Camera.import_scene_from_host_array(h_scene,h_sceneSize,32);
	Camera.import_lights_from_host(h_lights,h_lightsSize);

	Client client(1234,h_scene,h_sceneSize,&Camera); // networking object

	client.start();

	
	while(1) {
		if(Camera.frame_n % 500 == 0) {
			cout << "frame time: " << sum_time / 500 << " ms" << endl; // average frame time out of 500
			sum_time = 0;
		}


		while(client.generating_world) {
			cout << "loading world..." << endl;
		}; // wait while world loading

		while(SDL_PollEvent(&e)) {
			if(e.type == SDL_MOUSEWHEEL) {
				_inventory.moveSelect(e.wheel.y);
			}
			if(e.type == SDL_KEYDOWN) {
				if(e.key.keysym.scancode == SDL_SCANCODE_L) {
					move_light = !move_light;
					if(move_light) {
						current_light_index++;
						current_light_index = cycle(current_light_index,h_lightsSize);
					}
				}
				if(e.key.keysym.scancode == SDL_SCANCODE_H) {
					//Camera.ssaa = 16;
				}
				if(e.key.keysym.scancode < 40 && e.key.keysym.scancode > 29) {
					int numKey = e.key.keysym.scancode==39?0 :(e.key.keysym.scancode - 29);
					_inventory._items[_inventory.selected] = (items)numKey;
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
				if(e.button.button == SDL_BUTTON_RIGHT && _inventory.isBlock()) {
					vec3 w_n,_;
					int result_idx = Camera.tree.castRay(Camera.host_soa_scene,Camera.origin,Camera.direction,_,w_n,true);
					if(result_idx != (-1)) {
						client.place_block_net(Camera.host_soa_scene->_min[result_idx] + w_n,(blocks)(_inventory._items[_inventory.selected])); // says to the server i placed this block
					}
				}
				if(e.button.button == SDL_BUTTON_LEFT) {
					vec3 _;
					int result_idx = Camera.tree.castRay(Camera.host_soa_scene,Camera.origin,Camera.direction,_,_,true);
					if(result_idx != (-1)) {
						vec3 p = Camera.host_soa_scene->_min[result_idx];
						client.destroy_block_net(result_idx,p);
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
			curr_move_speed *= 1.3;
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

		if(client.updated_world) {
			Camera.import_scene_from_host_array(h_scene,h_sceneSize,32);
			client.updated_world = false;
		}

		Camera.render();

		_inventory.draw(&Camera);

		SDL_RenderPresent(Camera.sdl_renderer);

		if(Camera.frame_n > 1) {
			sum_time += Camera.frame_dt * 1000;
		}
	}
	client.stop();
}