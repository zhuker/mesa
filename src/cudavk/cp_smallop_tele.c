/*
 * The census table behind cp_smallop_tele.h. See that header for why the
 * counting happens at the call site rather than in a profiler.
 *
 * Sites are claimed lock-free out of a fixed table: a census that allocates
 * would perturb the thing it measures, and a census that locks would serialise
 * the submit thread against the recording thread. Slots are never freed, and
 * the report is printed at context teardown, after the threads have quiesced.
 */

#include "cp_smallop_tele.h"

#include <inttypes.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

bool cp_smallop_enabled = false;

#define CP_SMALLOP_SLOTS 512u

struct cp_smallop_slot {
   atomic_ullong key;      /* 0 while free */
   atomic_ullong count;
   atomic_ullong bytes;
   atomic_ullong small;    /* < 256 B */
   atomic_ullong mid;      /* < 4 KB */
   atomic_ullong big;      /* >= 4 KB */
   const char *file;
   const void *ra;
   int line;
   enum cp_smallop_kind kind;
};

static struct cp_smallop_slot cp_smallop_slots[CP_SMALLOP_SLOTS];
static atomic_ullong cp_smallop_lost;

static uint64_t
cp_smallop_key(const char *file, int line, const void *ra,
               enum cp_smallop_kind kind)
{
   uint64_t h = 1469598103934665603ull;
   h = (h ^ (uint64_t)(uintptr_t)file) * 1099511628211ull;
   h = (h ^ (uint64_t)(uint32_t)line) * 1099511628211ull;
   h = (h ^ (uint64_t)(uintptr_t)ra) * 1099511628211ull;
   h = (h ^ (uint64_t)kind) * 1099511628211ull;
   return h ? h : 1ull;
}

void
cp_smallop_note(const char *file, int line, const void *ra,
                enum cp_smallop_kind kind, size_t bytes)
{
   uint64_t key = cp_smallop_key(file, line, ra, kind);
   unsigned idx = (unsigned)(key % CP_SMALLOP_SLOTS);

   for (unsigned probe = 0; probe < CP_SMALLOP_SLOTS; probe++) {
      struct cp_smallop_slot *s =
         &cp_smallop_slots[(idx + probe) % CP_SMALLOP_SLOTS];
      uint64_t seen = atomic_load_explicit(&s->key, memory_order_acquire);
      if (!seen) {
         uint64_t expect = 0;
         if (!atomic_compare_exchange_strong_explicit(
                &s->key, &expect, key, memory_order_acq_rel,
                memory_order_acquire)) {
            if (expect != key)
               continue;
         } else {
            s->file = file;
            s->line = line;
            s->ra = ra;
            s->kind = kind;
         }
      } else if (seen != key) {
         continue;
      }

      atomic_fetch_add_explicit(&s->count, 1ull, memory_order_relaxed);
      atomic_fetch_add_explicit(&s->bytes, (unsigned long long)bytes,
                                memory_order_relaxed);
      if (bytes < 256)
         atomic_fetch_add_explicit(&s->small, 1ull, memory_order_relaxed);
      else if (bytes < 4096)
         atomic_fetch_add_explicit(&s->mid, 1ull, memory_order_relaxed);
      else
         atomic_fetch_add_explicit(&s->big, 1ull, memory_order_relaxed);
      return;
   }
   atomic_fetch_add_explicit(&cp_smallop_lost, 1ull, memory_order_relaxed);
}

static const char *
cp_smallop_kind_name(enum cp_smallop_kind kind)
{
   switch (kind) {
   case CP_SMALLOP_HTOD_ASYNC:   return "htod-async";
   case CP_SMALLOP_HTOD_SYNC:    return "htod-sync";
   case CP_SMALLOP_DTOH:         return "dtoh";
   case CP_SMALLOP_MEMSET_ASYNC: return "memset-async";
   case CP_SMALLOP_MEMSET_SYNC:  return "memset-sync";
   case CP_SMALLOP_CTXSYNC:      return "ctxsync";
   case CP_SMALLOP_UPLOAD_WRAP:  return "upload-wrap";
   case CP_SMALLOP_SUBMIT:       return "submit";
   default:                      return "?";
   }
}

static const char *
cp_smallop_base_name(const char *path)
{
   const char *slash = path ? strrchr(path, '/') : NULL;
   return slash ? slash + 1 : (path ? path : "?");
}

static int
cp_smallop_cmp(const void *a, const void *b)
{
   const struct cp_smallop_slot *const *x = a, *const *y = b;
   unsigned long long ca = atomic_load(&(*x)->count);
   unsigned long long cb = atomic_load(&(*y)->count);
   if (ca != cb)
      return ca < cb ? 1 : -1;
   return 0;
}

void
cp_smallop_report(void)
{
   if (!cp_smallop_enabled)
      return;

   struct cp_smallop_slot *live[CP_SMALLOP_SLOTS];
   unsigned n = 0;
   unsigned long long submits = 0;
   unsigned long long kind_count[CP_SMALLOP_KINDS] = { 0 };
   unsigned long long kind_bytes[CP_SMALLOP_KINDS] = { 0 };
   unsigned long long kind_small[CP_SMALLOP_KINDS] = { 0 };
   unsigned long long kind_mid[CP_SMALLOP_KINDS] = { 0 };
   unsigned long long kind_big[CP_SMALLOP_KINDS] = { 0 };

   for (unsigned i = 0; i < CP_SMALLOP_SLOTS; i++) {
      struct cp_smallop_slot *s = &cp_smallop_slots[i];
      if (!atomic_load(&s->key))
         continue;
      live[n++] = s;
      unsigned k = s->kind;
      kind_count[k] += atomic_load(&s->count);
      kind_bytes[k] += atomic_load(&s->bytes);
      kind_small[k] += atomic_load(&s->small);
      kind_mid[k] += atomic_load(&s->mid);
      kind_big[k] += atomic_load(&s->big);
      if (s->kind == CP_SMALLOP_SUBMIT)
         submits += atomic_load(&s->count);
   }

   /* These captures submit twice per frame; iteration 25's paired-submit
    * convention is the one every timing number here already uses. */
   double frames = submits ? (double)submits / 2.0 : 0.0;
   const void *base = NULL;
   Dl_info info;
   if (dladdr((const void *)cp_smallop_report, &info))
      base = info.dli_fbase;

   fprintf(stderr, "cudavk: smallops: submits=%llu frames=%.1f "
           "slots=%u lost=%llu base=%p\n", submits, frames, n,
           (unsigned long long)atomic_load(&cp_smallop_lost), base);

   for (unsigned k = 0; k < CP_SMALLOP_KINDS; k++) {
      if (!kind_count[k])
         continue;
      fprintf(stderr, "cudavk: smallops %-12s count=%llu (<256B=%llu "
              "<4K=%llu >=4K=%llu) bytes=%llu per_frame=%.2f\n",
              cp_smallop_kind_name(k), kind_count[k], kind_small[k],
              kind_mid[k], kind_big[k], kind_bytes[k],
              frames ? (double)kind_count[k] / frames : 0.0);
   }

   qsort(live, n, sizeof(live[0]), cp_smallop_cmp);
   for (unsigned i = 0; i < n; i++) {
      struct cp_smallop_slot *s = live[i];
      unsigned long long c = atomic_load(&s->count);
      if (!c)
         continue;
      fprintf(stderr, "cudavk: smallop site %-12s %s:%d",
              cp_smallop_kind_name(s->kind),
              cp_smallop_base_name(s->file), s->line);
      if (s->ra && base)
         fprintf(stderr, " caller+0x%tx", (const char *)s->ra -
                 (const char *)base);
      fprintf(stderr, " count=%llu per_frame=%.2f (<256B=%llu <4K=%llu "
              ">=4K=%llu) bytes=%llu\n", c,
              frames ? (double)c / frames : 0.0,
              (unsigned long long)atomic_load(&s->small),
              (unsigned long long)atomic_load(&s->mid),
              (unsigned long long)atomic_load(&s->big),
              (unsigned long long)atomic_load(&s->bytes));
   }
}
