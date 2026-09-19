/* Clip-counter seed bookkeeping; no CUDA operations or allocation policy. */
#ifndef CP_CLIP_SEED_H
#define CP_CLIP_SEED_H

#include <stdbool.h>
#include <stdint.h>

struct cp_clip_seed {
   uint64_t counter;
   uint32_t value;
};

static inline uint32_t
cp_clip_seed_value(bool stable, uint64_t active_ids, uint32_t max_clipped)
{
   return stable && !active_ids ? max_clipped : 0;
}

/* Record only a successfully scheduled launch with a thread zero. */
static inline void
cp_clip_seed_record(struct cp_clip_seed *seed, bool scheduled,
                    uint64_t counter, uint32_t value)
{
   if (scheduled && counter) {
      seed->counter = counter;
      seed->value = value;
   }
}

static inline bool
cp_clip_seed_needed(const struct cp_clip_seed *seed,
                    uint64_t counter, uint32_t value)
{
   return counter && (seed->counter != counter || seed->value != value);
}

#endif
