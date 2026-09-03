#ifndef CP_DEVOP_H
#define CP_DEVOP_H

/*
 * The device-operation epoch: how many stream operations that are not kernel
 * launches this process has enqueued.
 *
 * Two mechanisms need the same fact -- "nothing was issued on a stream between
 * these two points" -- and neither can get it by reading the source, because
 * the calls in question are spread over five files and one of them is issued
 * by cp_launch() itself.
 *
 *   - The programmatic-dependent-launch predecessor check in cp_launch_after()
 *     may only claim the attribute when the previous item on the stream really
 *     is the kernel the caller named. See cp_smallop_tele.h.
 *   - CUDAVK_VS_LANE runs a batch's vertex shader on a side stream, ordered
 *     against the main stream by an event recorded before the *previous*
 *     batch's chain. That is only sound if nothing else was enqueued in
 *     between, and this counter is how cp_vslane_take() checks it rather than
 *     asserting it.
 *
 * A kernel launch does not move the epoch: cp_launch() counts those in
 * cp->launches, and the PDL check wants a launch to be a legal predecessor.
 * Both readers therefore look at the pair.
 *
 * Unconditional -- one relaxed increment next to a call that costs about a
 * microsecond. It used to be armed by CUDAVK_PDL, which made it useless to
 * anything that has to be right when no debug flag is set.
 */

#include <stdint.h>

extern uint64_t cp_devop_epoch;

/* Every enqueue that is not a kernel launch: a clear, a copy, an event
 * record, a cross-stream wait, a host callback. */
static inline void
cp_devop_note(void)
{
   __atomic_fetch_add(&cp_devop_epoch, 1, __ATOMIC_RELAXED);
}

static inline uint64_t
cp_devop_now(void)
{
   return __atomic_load_n(&cp_devop_epoch, __ATOMIC_RELAXED);
}

#endif /* CP_DEVOP_H */
