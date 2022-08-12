#!/usr/bin/zsh
python3 -m grpc_tools.protoc --proto_path=../ --python_out=./python_gen --grpc_python_out=./python_gen ../gpu_profiling.proto
