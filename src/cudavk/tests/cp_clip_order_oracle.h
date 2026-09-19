/* Independent analytic oracle shared by the CPU and Vulkan ordering tests.
 * Geometry is in NDC, w=1, z=0.5. Red crosses x=-1; green is wholly inside.
 * Do not derive this oracle from compacted/output primitive IDs.
 */
#ifndef CP_CLIP_ORDER_ORACLE_H
#define CP_CLIP_ORDER_ORACLE_H
#include <math.h>
struct order_point { double x, y; };
static const struct order_point order_input[2][3] = {
   {{-2, -.75}, {.75, -.75}, {0, .75}},
   {{-.5, -.5}, {.5, -.5}, {0, .5}},
};
static double
order_edge(struct order_point a, struct order_point b, struct order_point p)
{
   return (b.x-a.x)*(p.y-a.y) - (b.y-a.y)*(p.x-a.x);
}
static int
order_contains(const struct order_point t[3], struct order_point p)
{
   return order_edge(t[0], t[1], p) >= 0 &&
          order_edge(t[1], t[2], p) >= 0 &&
          order_edge(t[2], t[0], p) >= 0;
}
/* Exclude samples near ORIGINAL edges, not implementation fan diagonals.
 * The oracle then does not depend on subpixel precision or fill conventions. */
static int
order_expected(unsigned x, unsigned y, unsigned w, unsigned h)
{
   struct order_point p = {2.0*(x+.5)/w-1, 2.0*(y+.5)/h-1};
   int winner = -1;
   for (int i = 0; i < 2; i++) {
      for (int e = 0; e < 3; e++) {
         struct order_point a = order_input[i][e], b = order_input[i][(e+1)%3];
         if (fabs(order_edge(a,b,p)) < .02 * hypot(b.x-a.x,b.y-a.y))
            return -2;
      }
      /* Both LEQUAL and GEQUAL accept equality: LAST INPUT wins. */
      if (order_contains(order_input[i],p)) winner = i;
   }
   return winner;
}
#endif
