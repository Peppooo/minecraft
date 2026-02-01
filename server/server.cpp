#include <enet/enet.h>
#include <stdio.h>
#include <iostream>
#include <unordered_set>
#include <vector>

using namespace std;

struct packet0 { // place block ch 0
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

struct packet1 { // destroy block ch 1
	int32_t index;
	int32_t x,y,z;
};

struct vec3 {
    float x,y,z;
};

typedef vec3 packet2; // player data recived from client

struct player {
    vec3 position;
    uint64_t identifier;
};

unordered_set<packet0,packet0hash> world;
bool broadcast_players_info = false;

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

    for(int i = -64; i <= 64; i++) {
        for(int j = -64; j <= 64; j++) {
            world.insert(packet0{i,0,j,6}); // grass block at i,0,j
        }
    }

    ENetEvent event;

 

    while(true) {
        // send player positions

        while(enet_host_service(server,&event,1)>0) {

            switch(event.type) {
                case ENET_EVENT_TYPE_CONNECT: {
                    broadcast_players_info = true;
                    printf("A new client connected from %x:%u.\n",event.peer->address.host,event.peer->address.port);
                    event.peer->data = new player;
                    (*(player*)event.peer->data).identifier = *((uint64_t*)&event.peer->address); // convert address bytes to 
                    vector<packet0> world_data(world.begin(),world.end());
                    auto packet = enet_packet_create(world_data.data(),world_data.size()*sizeof(packet0),ENET_PACKET_FLAG_RELIABLE);
                    if(enet_peer_send(event.peer,0,packet)) {
                        printf("unable to send world to connected peer\n");
                    }
                    break;
                }

                case ENET_EVENT_TYPE_RECEIVE: {
                    bool isPacketValid = false;
                    if(event.packet->dataLength <= max(max(sizeof(packet0),sizeof(packet1)),sizeof(packet2))) {
                        if(event.channelID == 0 && event.packet->dataLength == sizeof(packet0)) { // block placed
                            const packet0& packet = *((packet0*)event.packet->data);
                            if(world.find(packet) == world.end()) {
                                isPacketValid = true;
                                world.insert(packet);
                            }
        
                        }
                        else if(event.channelID == 1 && event.packet->dataLength == sizeof(packet1)) {
                            const packet1& packet = *((packet1*)event.packet->data);
                            packet0 real_pack = packet0{packet.x,packet.y,packet.z,0};
                            if(world.find(real_pack) != world.end()) {
                                isPacketValid = true;
                                world.erase(real_pack);
                            }
                            
                        }
                        else if(event.channelID == 2 && event.packet->dataLength == sizeof(packet2)) {
                            broadcast_players_info = true;
                            const packet2& data = *(packet2*)event.packet->data;
                            (*(player*)event.peer->data).position = data;
                            printf("new player position: %f %f %f\n",data.x,data.y,data.z);
                        }

                        if(isPacketValid) {
                            ENetPacket* packetCopy = enet_packet_create(event.packet->data, event.packet->dataLength, ENET_PACKET_FLAG_RELIABLE);

                            enet_host_broadcast(server,event.channelID,packetCopy);
                            enet_packet_destroy(event.packet);
                        }
                    }
                    break;
                }
                case ENET_EVENT_TYPE_DISCONNECT: {
                    printf("%llu disconnected.\n",(*(player*)event.peer->data).identifier);
                    break;
                }
                
            }
            if(broadcast_players_info) {

                for(int i = 0; i < server->connectedPeers; i++) { // send for each connected peer
                    int writes = 0;
                    packet2* personal_player_data = new packet2[server->connectedPeers-1];
                    for(int j = 0; j < server->connectedPeers; j++) {
                        if(i!=j) {
                            personal_player_data[writes] = (*(player*)server->peers[j].data).position;
                            writes++;
                        }
                    }
                    
                    enet_peer_send(&server->peers[i],2,enet_packet_create(personal_player_data,(server->connectedPeers - 1) * sizeof(packet2),NULL));
                    delete[] personal_player_data;
                }
                //enet_host_broadcast(server,2,enet_packet_create(player_data,server->connectedPeers*sizeof(packet2),NULL)); // send unreliable packet of players infos
                printf("sent players infos for %llu peers\n",server->connectedPeers);
                broadcast_players_info = false;
            }
        }

    }
    enet_host_destroy(server);
    enet_deinitialize();
    return 0;
}