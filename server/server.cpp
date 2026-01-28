#include <enet/enet.h>
#include <stdio.h>
#include <iostream>
#include <unordered_set>

using namespace std;

struct packet0 { // place block ch 0 / bytes: 13
	int32_t x,y,z; // 
	uint8_t block_type;
    bool operator==(const packet0& packet) const {
        return (packet.x == x && packet.y == y && packet.z == z);
    }
}; 

struct packet0hash {
	size_t operator()(const packet0& packet) const {
		size_t hx = hash<int>()(packet.x)
			  ,hy = hash<int>()(packet.y)
			  ,hz = hash<int>()(packet.z);
		return hx ^ (hy << 1) ^ (hz << 2);
	}
};

struct packet1 { // destroy block ch 1 / bytes: 16
	int32_t index;
	int32_t x,y,z;
};

unordered_set<packet0,packet0hash> world;

int _main() { // change for server compilation keep for client compilation
    if(enet_initialize() != 0) {
        printf("An error occurred while initializing ENet.\n");
        return 1;
        
    }

    ENetAddress address = {0};

    address.host = ENET_HOST_ANY; /* Bind the server to the default localhost.     */
    address.port = 1234;

    #define MAX_CLIENTS 32

    ENetHost* server = enet_host_create(&address,MAX_CLIENTS,3,0,0);

    if(server == NULL) {
        printf("An error occurred while trying to create an ENet server host.\n");
        return 1;
    }

    printf("Started server...\n");

    ENetEvent event;

    while(true) {

        while(enet_host_service(server,&event,1)>0) {

            switch(event.type) {
                case ENET_EVENT_TYPE_CONNECT: {
                    printf("A new client connected from %x:%u.\n",event.peer->address.host,event.peer->address.port);
                    vector<packet0> world_data(world.begin(),world.end());
                    auto packet = enet_packet_create(world_data.data(),world_data.size()*sizeof(packet0),ENET_PACKET_FLAG_RELIABLE);
                    if(enet_peer_send(event.peer,2,packet)) {
                        printf("unable to send world to connected peer\n");
                    }
                    break;
                }

                case ENET_EVENT_TYPE_RECEIVE: {
                    if(event.packet->dataLength <= max(sizeof(packet0),sizeof(packet1))) {
                        ENetPacket* packetCopy = enet_packet_create(
                            event.packet->data,
                            event.packet->dataLength,
                            ENET_PACKET_FLAG_RELIABLE
                        );
                        if(event.channelID == 0 && event.packet->dataLength == sizeof(packet0)) { // block placed
                            const packet0& packet = *((packet0*)event.packet->data);
                            if(world.find(packet) == world.end()) {
                                world.insert(packet);
                            }
        
                        }
                        else if(event.channelID == 1 && event.packet->dataLength == sizeof(packet1)) {
                            const packet1& packet = *((packet1*)event.packet->data);
                            packet0 real_pack = packet0{packet.x,packet.y,packet.z,0};
                            world.erase(real_pack);
                        }
                        enet_host_broadcast(server,event.channelID,packetCopy);
                        enet_packet_destroy(event.packet);
                    }
                    break;
                }
                case ENET_EVENT_TYPE_DISCONNECT: {
                    printf("%x:%u disconnected.\n",event.peer->address,event.peer->address.port);
                    break;
                }
                
            }
        }

    }
    enet_host_destroy(server);
    enet_deinitialize();
    return 0;
}