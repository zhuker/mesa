/*
 * Keep two native VkDevices and their CUDA contexts live while both execute
 * the blended A-buffer batch test, destroy them together, then create a third.
 * Each renderer must own all A-buffer allocations/events it launches against.
 *
 * Standalone build example:
 *   cc cpvk_multidevice.c -o cpvk_multidevice -lvulkan -lpthread
 */
#define _GNU_SOURCE
#define CPVK_BATCHBLEND_NO_MAIN
#include "cpvk_batchblend.c"

#include <pthread.h>
#include <stdbool.h>

struct thread_gate {
   pthread_barrier_t created;
   pthread_barrier_t finished;
};

struct thread_arg {
   struct thread_gate *gate;
   const char *out;
   int result;
};

static void
after_create(void *data)
{
   struct thread_arg *arg = data;
   pthread_barrier_wait(&arg->gate->created);
}

static void
before_destroy(void *data)
{
   struct thread_arg *arg = data;
   pthread_barrier_wait(&arg->gate->finished);
}

static void *
run_device(void *data)
{
   struct thread_arg *arg = data;
   char *argv[] = {
      (char *)"cpvk_multidevice",
      (char *)"/tmp/lat/tex.vert.spv",
      (char *)"/tmp/lat/tex.frag.spv",
      (char *)arg->out,
      NULL,
   };
   arg->result = cpvk_batchblend_run(4, argv, after_create, before_destroy, arg);
   return NULL;
}

static bool
files_equal(const char *a, const char *b)
{
   FILE *fa = fopen(a, "rb"), *fb = fopen(b, "rb");
   if (!fa || !fb) {
      if (fa) fclose(fa);
      if (fb) fclose(fb);
      return false;
   }

   bool equal = true;
   for (;;) {
      unsigned char aa[4096], bb[4096];
      size_t na = fread(aa, 1, sizeof(aa), fa);
      size_t nb = fread(bb, 1, sizeof(bb), fb);
      if (na != nb || memcmp(aa, bb, na)) {
         equal = false;
         break;
      }
      if (na < sizeof(aa))
         break;
   }
   fclose(fa);
   fclose(fb);
   return equal;
}

int
main(void)
{
   struct thread_gate gate;
   if (pthread_barrier_init(&gate.created, NULL, 2) ||
       pthread_barrier_init(&gate.finished, NULL, 2)) {
      fprintf(stderr, "barrier initialization failed\n");
      return 1;
   }

   struct thread_arg args[2] = {
      { .gate = &gate, .out = "/tmp/lat/cpvk_multidevice_0.ppm", .result = -1 },
      { .gate = &gate, .out = "/tmp/lat/cpvk_multidevice_1.ppm", .result = -1 },
   };
   pthread_t threads[2];
   if (pthread_create(&threads[0], NULL, run_device, &args[0]) ||
       pthread_create(&threads[1], NULL, run_device, &args[1])) {
      fprintf(stderr, "thread creation failed\n");
      return 1;
   }
   pthread_join(threads[0], NULL);
   pthread_join(threads[1], NULL);

   char *third_argv[] = {
      (char *)"cpvk_multidevice",
      (char *)"/tmp/lat/tex.vert.spv",
      (char *)"/tmp/lat/tex.frag.spv",
      (char *)"/tmp/lat/cpvk_multidevice_2.ppm",
      NULL,
   };
   int third = cpvk_batchblend_run(4, third_argv, NULL, NULL, NULL);

   bool equal = files_equal(args[0].out, args[1].out) &&
                files_equal(args[0].out, third_argv[3]);
   pthread_barrier_destroy(&gate.created);
   pthread_barrier_destroy(&gate.finished);

   if (args[0].result || args[1].result || third || !equal) {
      fprintf(stderr, "multidevice failed: devices=%d/%d third=%d equal=%d\n",
              args[0].result, args[1].result, third, equal);
      return 1;
   }
   printf("multidevice: three byte-identical A-buffer outputs\n");
   return 0;
}
