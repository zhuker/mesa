#!/bin/sh
set -eu
cd "$(dirname "$0")"
./optix_custom_bench ./device.optixir "${1:-5}" "${2:-16}"
