#include "texture.cuh"
#include "objects.cuh"
#include <vector>


enum blocks {
	dirt,
	cobblestone,
	brick,
	stone,
	sand,
	obsidian,
	grass,
	oak_planks,
	oak_log,
	quartz,
	count
};

vector<texture*> block_textures;
normal* default_normal_map;

void place_block(vec3 p,blocks block,cube* scene,size_t& sceneSize,material mat = material(diffuse)) {
	cube(p,p+vec3{1,1,1},scene,sceneSize,mat,block_textures[block],default_normal_map);
}

void load_textures(normal* _default_normal_map) {
	block_textures.resize(blocks::count);
	default_normal_map = _default_normal_map;
	for(int i = 0; i < blocks::count; i++) {
		char filename[128];
		snprintf(filename,128,"..\\textures\\%d.tex",i);
		IMPORT_TEXTURE(new_tex_p,filename,vec2(0,0),vec2(1,1),16,16);
		block_textures[i] = new_tex_p;
	}
}


class inventory {

};

class player {

};