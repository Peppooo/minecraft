cmake_minimum_required(VERSION 3.20)
project(networking)

add_executable(server server.cpp)

target_link_libraries(server
    PRIVATE
        enet
        winmm
        ws2_32
        iphlpapi
)