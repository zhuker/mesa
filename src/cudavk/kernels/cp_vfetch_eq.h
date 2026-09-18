/* Validation-only fetch ABI. No production offsets are changed. */
#ifndef CP_VFETCH_EQ_H
#define CP_VFETCH_EQ_H
#ifndef __CUDACC_RTC__
#include <stdint.h>
#endif
struct cp_vfetch_eq_row {
   uint32_t valid, vertex_id, instance_id, row;
   uint32_t slots[32][4];
};
struct cp_vfetch_eq_report {
   uint64_t invocations, valid, slots, words;
   uint64_t diff_validity, diff_ids, diff_slots, diff_words;
   uint64_t differing_invocations, injections;
   uint64_t first_claim, first_dispatch, first_invocation, first_slot;
   uint64_t first_word, first_expected, first_actual;
};
#endif
