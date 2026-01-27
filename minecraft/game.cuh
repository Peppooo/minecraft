#pragma once
#include "texture.cuh"
#include "objects.cuh"
#include <SDL2/SDL_image.h>
#include <vector>


enum blocks {
	dirt_block,
	cobblestone_block,
	brick_block,
	stone_block,
	sand_block,
	obsidian_block,
	grass_block,
	oak_planks_block,
	oak_log_block,
	quartz_block,
	glow_stone_block,

	blocks_count
};

enum items {
	dirt_item,
	cobblestone_item,
	brick_item,
	stone_item,
	sand_item,
	obsidian_item,
	grass_item,
	oak_planks_item,
	oak_log_item,
	quartz_item,
	glowstone_block,

	items_count,
	empty_item
};

enum gui_textures {
	gui_bar,
	gui_selected_bar,

	gui_count
};

vector<texture*> block_textures;
vector<SDL_Texture*> items_textures;
vector<SDL_Texture*> gui_textures;

normal* default_normal_map;

void place_block(vec3 p,blocks block,cube* scene,size_t& sceneSize,material mat = material(diffuse)) {
	if(block == quartz_block || block==obsidian_block) mat = material(specular);
	if(block == glowstone_block) mat.emission = 10;
	cube(p,p+vec3{1,1,1},scene,sceneSize,mat,block_textures[block],default_normal_map);

}

void load_textures(SDL_Renderer* sdl_ren,normal* _default_normal_map) {
	IMG_Init(IMG_INIT_PNG);
	block_textures.resize(blocks_count);
	default_normal_map = _default_normal_map;
	char filename[64];
	for(int i = 0; i < blocks_count; i++) {
		snprintf(filename,64,"..\\textures\\blocks\\%d.tex",i);
		IMPORT_TEXTURE(new_tex_p,filename,vec2(0,0),vec2(1,1),16,16);
		block_textures[i] = new_tex_p;
	}
	items_textures.resize(items_count);
	for(int i = 0; i < items_count; i++) {
		snprintf(filename,64,"..\\textures\\items\\%d.png",i);
		items_textures[i] = IMG_LoadTexture(sdl_ren,filename);
		if(!items_textures[i]) printf("unable to load item texture n.%d, error: %s\n",i,IMG_GetError());
	}
	gui_textures.resize(gui_count);
	for(int i = 0; i < gui_count; i++) {
		snprintf(filename,64,"..\\textures\\gui\\%d.png",i);
		gui_textures[i] = IMG_LoadTexture(sdl_ren,filename);
		if(!gui_textures[i]) printf("unable to load gui texture n.%d, error: %s\n",i,IMG_GetError());
	}
}


class inventory {
public:
	uint8_t selected; // selected item from 0 -> 9
	vector<items> _items;

	inventory() {
		_items = vector<items>(9,items::empty_item);
	}
	bool isBlock() {
		return (_items[selected] < blocks_count);
	}
	void moveSelect(int8_t dir) {
		selected = ((90+(int)selected-dir) % 9);
	}
	void draw(renderer* ren) {
		double block_resize_fact = 0.7;
		int quad_size = ren->w/13;
		SDL_Rect sel_rect = {ren->w/2-(quad_size*4.5),ren->h - quad_size,quad_size,quad_size};

		SDL_Rect bar_rect = {ren->w/2-(quad_size*4.5),ren->h - quad_size,quad_size*9,quad_size};

		SDL_RenderCopy(ren->sdl_renderer,gui_textures[gui_bar],NULL,&bar_rect);

		
		for(int i = 0; i < 9; i++) {
			
			vec2 sel_center(sel_rect.x+sel_rect.w/2,sel_rect.y+sel_rect.h/2);
			
			SDL_Rect block_rect = {sel_rect.x,sel_rect.y,quad_size*block_resize_fact,quad_size*block_resize_fact};

			block_rect.x = sel_center.x-block_rect.w *0.5;
			block_rect.y = sel_center.y-block_rect.w *0.5;

			if(_items[i] != empty_item) {
				SDL_RenderCopy(ren->sdl_renderer,items_textures[_items[i]],NULL,&block_rect);
			}
			if(i == selected) {
				SDL_RenderCopy(ren->sdl_renderer,gui_textures[gui_selected_bar],NULL,&sel_rect);
			}
			sel_rect.x += sel_rect.w;
		}

	}
};