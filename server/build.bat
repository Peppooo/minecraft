cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
cd ..
build\Release\server.exe