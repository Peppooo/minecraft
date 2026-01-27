#include <enet/enet.h>
#include <stdio.h>

int main() {
    if(enet_initialize() != 0) {
        printf("An error occurred while initializing ENet.\n");
        return 1;
    }

    ENetAddress address = {0};

    address.host = ENET_HOST_ANY; /* Bind the server to the default localhost.     */
    address.port = 1234; /* Bind the server to port 7777. */

    #define MAX_CLIENTS 32

    /* create a server */
    ENetHost* server = enet_host_create(&address,MAX_CLIENTS,2,0,0);

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
                    break;
                }

                case ENET_EVENT_TYPE_RECEIVE: {
                    printf("A packet of length %lu containing %s was received from %s on channel %u.\n",
                        event.packet->dataLength,
                        event.packet->data,
                        event.peer->data,
                        event.channelID);
                    ENetPacket* packetCopy = enet_packet_create(
                        event.packet->data,
                        event.packet->dataLength,
                        ENET_PACKET_FLAG_RELIABLE
                    );
                    enet_host_broadcast(server,0,packetCopy);
                    enet_packet_destroy(event.packet);
                    break;
                }
                case ENET_EVENT_TYPE_DISCONNECT: {
                    printf("%s disconnected.\n",event.peer->data);
                    event.peer->data = NULL;
                    break;
                }
                
            }
        }

    }
    enet_host_destroy(server);
    enet_deinitialize();
    return 0;
}