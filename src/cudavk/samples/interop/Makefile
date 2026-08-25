# Builds the Vulkan producer half of the CUDA/PyTorch interop sample.
#
#   make            build cpvk_interop_render and both .spv shaders
#   make self-test  run the producer against its own fake-consumer mode
#   make clean

CC       ?= gcc
CFLAGS   ?= -O2 -g -Wall -Wextra -Wno-unused-parameter -std=gnu11
LDLIBS   := -lvulkan -lz -lm
GLSLANG  ?= glslangValidator

BIN     := cpvk_interop_render
SPV     := interop.vert.spv interop.frag.spv

all: $(BIN) $(SPV)

$(BIN): cpvk_interop_render.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

%.spv: %
	$(GLSLANG) -V $< -o $@

interop.vert.spv: interop.vert
interop.frag.spv: interop.frag

self-test: all
	./$(BIN) --self-test --frames 8 --slots 3 --width 128 --height 128 --sync timeline

clean:
	rm -f $(BIN) $(SPV)

.PHONY: all clean self-test
