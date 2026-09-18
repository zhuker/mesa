/* Stable host issue ordering only; segment identity and ownership never move. */
#ifndef CP_RUN_ORDER_H
#define CP_RUN_ORDER_H

#include <stdbool.h>
#include <stdint.h>

static inline void
cp_run_order(unsigned *order, const uint32_t *triangles, unsigned n,
             bool largest)
{
   for (unsigned i = 0; i < n; i++) {
      unsigned j = i;
      /* Strict comparison retains submission order for equal scores. Triangle
       * lists execute three vertices per input triangle, so this also orders
       * the exact preparation's vertex work without a capture-specific model. */
      while (largest && j && triangles[order[j - 1]] < triangles[i]) {
         order[j] = order[j - 1];
         j--;
      }
      order[j] = i;
   }
}

#endif
