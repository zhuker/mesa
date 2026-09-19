/* CPU-only: cc -std=c11 -Wall -Wextra -Werror -Isrc/cudavk
 * src/cudavk/tests/cp_clip_seed_test.c -o /tmp/cp_clip_seed_test
 * /tmp/cp_clip_seed_test
 * Mock allocation/launch ordering, using the production seed decisions.
 * Does not simulate CUDA execution or establish image correctness. */
#include "cp_clip_seed.h"
#include <assert.h>
#include <stdio.h>

struct mock {
   unsigned counter_calls, active_calls, seeds;
   unsigned tick, vs_done, seed_at, consume_at;
   uint32_t memory;
};

static uint64_t
alloc(struct mock *m, bool active, bool succeeds)
{
   if (active)
      m->active_calls++;
   else
      m->counter_calls++;
   return succeeds ? (active ? 0x2000 : 0x1000) : 0;
}

static void
run(bool stable, bool fold, bool fused, bool hoist,
    bool early_counter_ok, bool early_active_ok, bool retry_ok,
    bool launch_ok, bool skip_seed)
{
   const uint32_t max_clipped = 80;
   struct mock m = {0};
   struct cp_clip_seed seeded = {0};
   uint64_t early_counter = fold ? alloc(&m, false, early_counter_ok) : 0;
   uint64_t early_active = fold && stable ? alloc(&m, true, early_active_ok) : 0;
   uint32_t early_value = cp_clip_seed_value(stable, early_active, max_clipped);
   bool writes = fold && launch_ok && !skip_seed && early_counter;
   if (!fused) {
      cp_clip_seed_record(&seeded, writes, early_counter, early_value);
      if (writes) { m.seeds++; m.memory = early_value; }
   }

   uint64_t counter = 0, active = 0;
   /* Exactly one final allocation decision: before VS if hoisted, after
    * VS otherwise. A successful early allocation must never be repeated. */
   for (unsigned phase = 0; phase < 2; phase++) {
      if (phase == (hoist ? 0u : 1u)) {
         counter = early_counter ? early_counter : alloc(&m, false, retry_ok);
         if (counter)
            active = early_active ? early_active :
                     stable ? alloc(&m, true, retry_ok) : 0;
      }
      if (!phase) {
         if (fused) {
            cp_clip_seed_record(&seeded, writes, early_counter, early_value);
            if (writes) { m.seeds++; m.memory = early_value; }
         }
         m.vs_done = ++m.tick; /* Includes lane join, if present. */
      }
   }
   uint32_t value = cp_clip_seed_value(stable, active, max_clipped);
   bool repair = !(fold && skip_seed) &&
                 cp_clip_seed_needed(&seeded, counter, value);
   if (repair) {
      m.seed_at = ++m.tick;
      m.seeds++;
      m.memory = value;
      cp_clip_seed_record(&seeded, true, counter, value);
      assert(!cp_clip_seed_needed(&seeded, counter, value));
   }
   if (counter) {
      m.consume_at = ++m.tick;
      if (!(fold && skip_seed)) assert(m.memory == value);
      if (repair) assert(m.vs_done < m.seed_at && m.seed_at < m.consume_at);
   } /* No counter is missing storage, NOT a correct unclipped fallback. */
   assert(m.counter_calls == (unsigned)(fold ? (early_counter_ok ? 1 : 2) : 1));
   assert(m.active_calls == (unsigned)(stable ?
          ((fold ? 1 : 0) + (counter && !early_active ? 1 : 0)) : 0));
   assert(m.seeds == (unsigned)writes + (unsigned)repair);
   if (writes && counter == early_counter && value == early_value)
      assert(!repair); /* Normal path: no added operation. */
}

/* An explicit dependency model, NOT execution of the renderer. The source
 * audit establishes that these modeled edges are actually issued. */
enum stream_mode { MAIN, VERTEX_LANE, OPAQUE_SEGMENT };
static void
schedule(enum stream_mode mode, bool pdl, bool deferred, bool repair)
{
   unsigned main_tail = 0, lane_tail = 0, segment_tail = 0;
   unsigned *current = mode == OPAQUE_SEGMENT ? &segment_tail : &main_tail;
   unsigned *vs_stream = mode == VERTEX_LANE ? &lane_tail : current;
   unsigned seed_write = ++*vs_stream;
   unsigned vs_complete = ++*vs_stream;
   unsigned event_record = 0, event_wait = 0;
   if (mode == VERTEX_LANE) {
      event_record = ++lane_tail;
      main_tail = event_record + 1; /* wait edge */
      event_wait = main_tail;
      assert(vs_complete < event_record && event_record < event_wait);
   }
   /* Memsets have normal stream order, not PDL launch attributes. Their
    * devop epoch also invalidates the preceding-kernel PDL token. */
   unsigned epoch = 0, predecessor_epoch = 0;
   unsigned repair_write = repair ? (++epoch, ++*current) : 0;
   bool pdl_allowed = pdl && epoch == predecessor_epoch;
   if (repair) assert(!pdl_allowed);
   if (deferred) ++*current; /* unrelated work may delay, never advance it */
   unsigned consumer_read = ++*current;
   assert(seed_write < vs_complete && vs_complete < consumer_read);
   if (repair) assert(vs_complete < repair_write && repair_write < consumer_read);
   if (event_wait) assert(event_wait < consumer_read);
   if (mode == OPAQUE_SEGMENT) assert(!main_tail && !lane_tail);
   /* A PDL-admitted consumer waits for producer completion before reading.
    * This model intentionally provides no cross-stream implicit ordering. */
   if (pdl_allowed) assert(vs_complete < consumer_read);
}

int main(void)
{
   /* All allocation outcomes, 0/max seeds, hoisted/non-hoisted, classic/
    * fused VF, folded/unfolded (including vertex-lane), and fault control. */
   for (unsigned bits = 0; bits < 512; bits++)
      run(bits & 1, bits & 2, bits & 4, bits & 8, bits & 16,
          bits & 32, bits & 64, bits & 128, bits & 256);
   for (unsigned mode = MAIN; mode <= OPAQUE_SEGMENT; mode++)
      for (unsigned bits = 0; bits < 8; bits++)
         schedule(mode, bits & 1, bits & 2, bits & 4);
   struct cp_clip_seed s = {0};
   cp_clip_seed_record(&s, false, 1, 0);
   assert(cp_clip_seed_needed(&s, 1, 0));
   cp_clip_seed_record(&s, true, 1, 0);
   assert(cp_clip_seed_needed(&s, 2, 0)); /* Pointer change. */
   assert(cp_clip_seed_needed(&s, 1, 80)); /* Mode change. */
   assert(!cp_clip_seed_needed(&s, 0, 80)); /* Failed retry. */
   puts("clip seed tests: PASS");
}
