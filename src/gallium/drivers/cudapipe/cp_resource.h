#ifndef CP_RESOURCE_H
#define CP_RESOURCE_H

struct pipe_screen;
struct pipe_context;

void cudapipe_init_screen_resource_funcs(struct pipe_screen *screen);
void cudapipe_init_context_resource_funcs(struct pipe_context *ctx);

#endif
