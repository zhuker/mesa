/* Static whole-segment placement. No device state or completion estimates. */
#ifndef CP_PASS_SCHEDULE_H
#define CP_PASS_SCHEDULE_H

#include <stdint.h>

/* Fill the prefix before balancing, so prefix-based joins remain valid.
 * Ties retain the original round-robin slot for deterministic placement. */
static inline unsigned
cp_pass_schedule(uint64_t *load, unsigned slots, unsigned ordinal,
                 uint64_t cost)
{
   unsigned k = ordinal % slots;
   if (ordinal >= slots) {
      for (unsigned i = 0; i < slots; i++)
         if (load[i] < load[k])
            k = i;
   }
   /* Saturate rather than wrap an accumulated estimate. */
   load[k] = cost > UINT64_MAX - load[k] ? UINT64_MAX : load[k] + cost;
   return k;
}

#endif
