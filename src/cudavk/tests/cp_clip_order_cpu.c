/* CPU-only synthetic clip ordering model, NOT a driver execution test.
 * cc -std=c11 -Wall -Wextra -Werror cp_clip_order_cpu.c -lm -o /tmp/clip-order-cpu
 * Simulate all arrival orders, including green allocating before clipped red.
 */
#include <stdio.h>
#include "cp_clip_order_oracle.h"

int main(void)
{
   /* Exact analytic x=-1 intersections of the red triangle: (-1,0),
    * (-1,-.75). This produces two fan children; green produces one. */
   const struct order_point fan[3][3] = {
      {{-1,0},{-1,-.75},{.75,-.75}},
      {{-1,0},{.75,-.75},{0,.75}},
      {{-.5,-.5},{.5,-.5},{0,.5}},
   };
   const int input_id[3] = {0,0,1};
   const int arrivals[6][3] = {
      {0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}
   };
   unsigned counts[3] = {0}, wrong_allocation = 0;
   for (int mode = 0; mode < 2; mode++) /* LEQUAL and GEQUAL, same z */
      for (int a = 0; a < 6; a++)
         for (unsigned y = 0; y < 64; y++)
            for (unsigned x = 0; x < 64; x++) {
               int expected = order_expected(x,y,64,64);
               if (expected == -2) continue;
               struct order_point p = {2.0*(x+.5)/64-1,2.0*(y+.5)/64-1};
               int stable = -1, allocated = -1;
               for (int slot = 0; slot < 3; slot++) {
                  int f = arrivals[a][slot];
                  if (!order_contains(fan[f],p)) continue;
                  if (input_id[f] > stable) stable = input_id[f];
                  allocated = input_id[f]; /* broken: last allocation wins */
               }
               if (stable != expected) {
                  fprintf(stderr,"mode=%d arrival=%d pixel=%u,%u expected=%d got=%d\n",
                          mode,a,x,y,expected,stable);
                  return 1;
               }
               counts[expected+1]++;
               if (allocated != expected) wrong_allocation++;
            }
   if (!counts[0] || !counts[1] || !counts[2] || !wrong_allocation) return 1;
   /* Admission negative controls: stable input IDs repair only ties. A
    * depth-first visibility key is not last-input semantics when depth testing
    * is disabled, or when inclusive depth testing does NOT write depth.
    * Clear=1 for LEQUAL (0 for GEQUAL): both differently-depth fragments pass
    * independently with writes disabled, so green must still win. */
   unsigned policy_counterexamples = 0;
   for (int test_enabled = 0; test_enabled <= 1; test_enabled++)
      for (int greater = 0; greater <= 1; greater++) {
         const double z[2] = {greater ? .75 : .25, greater ? .25 : .75};
         const double clear = greater ? 0 : 1;
         int sequential = -1, depth_key = -1;
         for (int input = 0; input < 2; input++) {
            int passes = !test_enabled ||
                         (greater ? z[input] >= clear : z[input] <= clear);
            if (!passes) continue;
            sequential = input; /* no depth write: compare still uses clear */
            if (depth_key < 0 ||
                (greater ? z[input] > z[depth_key] : z[input] < z[depth_key]) ||
                (z[input] == z[depth_key] && input > depth_key))
               depth_key = input;
         }
         if (sequential != 1 || depth_key != 0) return 1;
         policy_counterexamples++;
      }
   printf("clip order CPU: PASS background=%u red=%u green=%u allocation-negative=%u "
          "unsupported-depth-policy-counterexamples=%u\n",
          counts[0],counts[1],counts[2],wrong_allocation,policy_counterexamples);
   return 0;
}
