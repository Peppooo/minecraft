#pragma once
#include <enet\enet.h>
#include <thread>
#include "game.cuh"
#include "engine.cuh"
#include <unordered_set>

struct packet0 { // place block ch 0
	int32_t x,y,z; // 
	uint8_t block_type;
};

struct packet1 { // destroy block ch 1
	int32_t index;
	int32_t x,y,z;
};

typedef vec3 packet2; // player status ch 2 


class Client {
public:
	// networking
	ENetHost* client = nullptr;
	ENetPeer* peer = nullptr;
	ENetEvent net_event;
	bool generating_world = false;

	// threads
	thread t_event_handle,t_status_handle;
	bool running_handle = true;

	// scene
	cube* scene;
	size_t* sceneSize;
	renderer* ren;

	vec3* player_position;

	bool updated_world = false;

	uint64_t tick = 0;

	Client(uint16_t port,cube* _scene,size_t& _sceneSize,renderer* _ren,vec3* _player_position = nullptr):scene(_scene),ren(_ren) {
		if(enet_initialize() != 0) {
			printf("cannot initialize enet\n");
		}
		if(!_player_position) {
			player_position = &(_ren->origin);
		}
		else {
			player_position = _player_position;
		}
		sceneSize = &_sceneSize;
		client = enet_host_create(nullptr,1,0,0,0); // peerCount
		if(!client) printf("cannot create enet host\n");
		ENetAddress addr;
		enet_address_set_host(&addr,ENET_HOST_ANY); // "92.4.166.112"
		addr.port = port;
		peer = enet_host_connect(client,&addr,3,0); // address, channels count
		if(!peer) printf("cannot connect with server\n");
	}
	void place_block_net(const vec3& p,const blocks& block) {
		packet0 packet_data = {int32_t(p.x),int32_t(p.y),int32_t(p.z),uint8_t(block)};

		auto packet = enet_packet_create(&packet_data,sizeof(packet0),ENET_PACKET_FLAG_RELIABLE);

		if(enet_peer_send(peer,0,packet)) printf("Cannot send block placing packet\n");// send on channel 0
	}
	void destroy_block_net(const int32_t idx,const vec3& Min) {
		packet1 packet_data = {idx,Min.x,Min.y,Min.z};

		auto packet = enet_packet_create(&packet_data,sizeof(packet1),ENET_PACKET_FLAG_RELIABLE);

		if(enet_peer_send(peer,1,packet)) printf("Cannot send block destroy packet\n");// send on channel 1

	}
	void status_handler() {
		tick = 0;
		while(running_handle) {
			vec3 data = *player_position;
			auto packet = enet_packet_create(&data,sizeof(packet2),0); // unreliable packet
			enet_peer_send(peer,2,packet);

			this_thread::sleep_for(chrono::milliseconds(32)); // ticks 32 milliseconds
			tick++;
		}
	}
	void event_handler() {
		bool _world_update = false;
		while(running_handle) {

			while(enet_host_service(client,&net_event,1) > 0) {
				if(net_event.type == ENET_EVENT_TYPE_CONNECT) {}
				if(net_event.type == ENET_EVENT_TYPE_DISCONNECT) {
					printf("disconnected from the server.\n");
					running_handle = false;
				};

				if(net_event.type == ENET_EVENT_TYPE_RECEIVE) {
					if(net_event.channelID == 0) { // channel 0 place block
						packet0* data = (packet0*)net_event.packet->data;
						if(net_event.packet->dataLength > sizeof(packet0)) { // 3 4 byte integers for block position + 1 byte block type
							printf("loading world (dataLen:%llu) from server... ",net_event.packet->dataLength/sizeof(packet0));
							generating_world = true;
							packet0* data = (packet0*)net_event.packet->data;
							for(int i = 0; i < (net_event.packet->dataLength / sizeof(packet0)); i++) {
								place_block({float(data[i].x),float(data[i].y),float(data[i].z)},(blocks)data[i].block_type,scene,*sceneSize);
							}
							generating_world = false;
							_world_update = true;
							printf("done\n");
						}
						else if(net_event.packet->dataLength == sizeof(packet0)) {
							printf("recived valid new block placing from server.\n");
							place_block(vec3{(float)data->x,(float)data->y,(float)data->z},(blocks)data->block_type,scene,*sceneSize);
							_world_update = true;
						}
						else {
							printf("invalid packet0 recived\n");
						}
					}
					else if(net_event.channelID == 1) { // channel 1 destroy block
						packet1* data = (packet1*)net_event.packet->data;
						if(net_event.packet->dataLength != sizeof(packet1)) { // 3 4 byte integers for block position + 1 byte block type
							printf("invalid data from the server\n");
						}
						else {
							printf("recived valid block destroy from server.\n");
							const vec3& m = ren->host_soa_scene->_min[data->index];
							const vec3 target = vec3{(float)data->x,(float)data->y,(float)data->z};
							if(!(m == target)) {
								printf("INVALID INDEX DESTROY %f %f %f != %f %f %f\n",(float)data->x,(float)data->y,(float)data->z,m.x,m.y,m.z);
								for(int i = 0; i < *sceneSize; i++) {
									if(scene[i]._min == target) {
										swap(scene[i],scene[*sceneSize - 1]);
										(*sceneSize)--;
									}
								}
							}
							else {
								swap(scene[data->index],scene[*sceneSize-1]);
								(*sceneSize)--;
							}
							_world_update = true;
						}
					}
				}
			}

			updated_world = _world_update;
		}
	}
	void start() {
		t_event_handle = thread(&Client::event_handler,this);
		t_status_handle = thread(&Client::status_handler,this);
	}
	void stop() {
		running_handle = false;
		if(t_event_handle.joinable()) t_event_handle.join();
		if(t_status_handle.joinable()) t_status_handle.join();
	}
	~Client() {
		stop();
		enet_host_destroy(client);
	}
};