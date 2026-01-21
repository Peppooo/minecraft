#pragma once
#include "objects.cuh"

constexpr int w = 1920,h = 1080;
constexpr float fov = M_PI / 1.6f;
constexpr float move_speed = 2.0f;
constexpr float mouse_sens = 0.001f;
float foc_len = w / (2 * tanf(fov / 2.0f));
bool hq = false;

vec3 origin = {0,2,0};
bool move_light = false;
int current_light_index = 0;
float yaw = 0.0f,pitch = 0.0f,roll = 0.0f;

int wid_Size = 6;