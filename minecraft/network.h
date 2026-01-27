#pragma once
#include <enet\enet.h>
#include <thread>
#include "game.cuh"

class Client {
public:
	// networking
	ENetHost* client = nullptr;
	ENetPeer* peer = nullptr;

	ENetEvent net_event;

	// threads

	thread t_event_handle;
	bool running_handle = true;

	// scene
	cube* scene;
	size_t* sceneSize;

	renderer* ren;

	Client(uint16_t port,cube* _scene,size_t& _sceneSize,renderer* _ren):scene(_scene),ren(_ren) {
		if(enet_initialize() != 0) {
			perror("cannot initialize enet\n");
		}
		sceneSize = &_sceneSize;
		client = enet_host_create(nullptr,1,2,0,0); // peerCount,channels,
		if(!client) perror("cannot create enet host\n");
		ENetAddress addr;
		enet_address_set_host(&addr,ENET_HOST_ANY);
		addr.port = port;
		peer = enet_host_connect(client,&addr,2,0);
		if(!peer) perror("cannot connect with server\n");
	}
	void place_block_net(const vec3& p,const blocks& block) {
		uint8_t* data = (uint8_t*)malloc(sizeof(int)*3+1);
		((int32_t*)data)[0] = int32_t(p.x);
		((int32_t*)data)[1] = int32_t(p.y);
		((int32_t*)data)[2] = int32_t(p.z);
		data[12] = (uint8_t)block;
		auto pack = enet_packet_create(data,sizeof(int32_t) * 3 + 1,ENET_PACKET_FLAG_RELIABLE);
		enet_peer_send(peer,0,pack);
	}
	void event_handler() {
		bool placed_blocks = false;
		while(running_handle) {


			while(enet_host_service(client,&net_event,50) > 0) {
				if(net_event.type == ENET_EVENT_TYPE_CONNECT) printf("connected to server.\n");
				if(net_event.type == ENET_EVENT_TYPE_DISCONNECT) {
					printf("disconnected from the server.\n");
					running_handle = false;
				};

				if(net_event.type == ENET_EVENT_TYPE_RECEIVE) {
					int* data = (int*)net_event.packet->data;
					if(net_event.packet->dataLength != (sizeof(int) * 3 + 1)) { // 3 4 byte integers for block position + 1 byte block type
						printf("invalid data from the server\n");
					}
					else {
						placed_blocks = true;
						printf("recived valid new block placing from server.\n");
						place_block(vec3{(float)data[0],(float)data[1],(float)data[2]},
							(blocks)net_event.packet->data[12],scene,*sceneSize);
					}
				}
			}
			if(placed_blocks) {
				ren->import_scene_from_host_array(scene,*sceneSize,32);
				placed_blocks = false;
			}
		}
	}
	void start() {
		t_event_handle = thread(&Client::event_handler,this);
	}
	void stop() {
		running_handle = false;
		if(t_event_handle.joinable()) t_event_handle.join();
	}
	~Client() {
		stop();
		enet_host_destroy(client);
	}
};