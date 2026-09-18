#!/bin/sh
set -eu
cd "$(dirname "$0")"
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
OPTIX=${OPTIX:-/tmp/optix-headers}
"$NVCC" -std=c++17 -O3 --optix-ir -I"$OPTIX" -o device.optixir device.cu
"$NVCC" -std=c++17 -O3 -I"$OPTIX" -o optix_custom_bench bench.cpp -ldl -lcuda
${CXX:-c++} -std=c++17 -O3 -I"$OPTIX" -I/usr/local/cuda/include \
  -o optix_bvh_bench cp_optix_bvh_bench.cpp -ldl -lcuda
