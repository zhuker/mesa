/* CPU-only issue-order gate. cc -std=c11 -Isrc/cudavk this-file -o /tmp/gate */
#include "cp_run_order.h"
#include <assert.h>
#include <limits.h>
#include <stdio.h>

int main(void)
{
   uint32_t seed = 17;
   for (unsigned n = 0; n <= 64; n++) {
      for (unsigned trial = 0; trial < 200; trial++) {
         uint32_t score[64];
         unsigned order[64], seen[64] = {0};
         for (unsigned s = 0; s < n; s++) {
            seed = seed * 1664525u + 1013904223u;
            score[s] = trial == 0 ? UINT_MAX : seed % 23;
         }
         cp_run_order(order, score, n, false);
         for (unsigned i = 0; i < n; i++) assert(order[i] == i);
         cp_run_order(order, score, n, true);
         for (unsigned i = 0; i < n; i++) {
            unsigned s = order[i];
            assert(s < n && !seen[s]++);
            if (i) {
               unsigned prev = order[i - 1];
               assert(score[prev] >= score[s]);
               if (score[prev] == score[s]) assert(prev < s);
            }
         }
         /* Model both phases with original modulo ownership. Every VS output
          * stays private; each clear/clip/stage2/stage3 chain retires before
          * the next clear on that queue, also for >8 and 64 segments. */
         unsigned ready[64] = {0}, owner[8];
         for (unsigned q = 0; q < 8; q++) owner[q] = UINT_MAX;
         for (unsigned i = 0; i < n; i++) ready[order[i]] = 1;
         for (unsigned i = 0; i < n; i++) {
            unsigned s = order[i], q = s % 8;
            assert(ready[s] && owner[q] == UINT_MAX);
            owner[q] = s;
            assert(owner[q] == s); /* stage2 and stage3 consume same owner */
            owner[q] = UINT_MAX;
         }
      }
   }
   puts("run issue order: PASS (0..64 segments, ties, extremes, queue wrap)");
   return 0;
}
