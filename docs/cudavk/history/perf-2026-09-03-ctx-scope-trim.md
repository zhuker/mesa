# Trimming the context scope from the record-only entry points

Every entry point in this driver opens with `CPVK_CTX_SCOPE`, which pushes the
device's CUDA context and pops it on the way out. That is correct where a CUDA
call is reachable, because every call the driver makes acts on whatever context
is current on the *application's* thread. It is two driver calls for nothing
where no CUDA call is reachable, and most of the command-recording surface is
exactly that: a `vkCmdDraw` copies state into `cmd->ops` and returns. The
analysis of 2026-09-02 (lead G) counted about **900 push/pop pairs per heavy
frame** on entry points of that kind and measured 0.07 ms/frame of host time on
a frame that is host-bound.

`CPVK_CTX_SCOPE_RECORD` is the same scope, made conditional: it pushes only
under `CUDAVK_NO_CTX_SCOPE_TRIM=1`, which restores the old sequence exactly.
`cpvk_ctx_leave` already pops only what was pushed, so the trimmed arm needs no
second code path.

## Which entry points, and how "no CUDA call" was decided

Reachability, not the function body. The call graph of every function defined
in `src/cudavk/*.c` and `src/cudavk/*.h` was closed over its callees and any
function containing a `cu*` driver-API call was treated as tainted; the closure
was deliberately over-approximated (parse artefacts add callees rather than
drop them). 36 entry points in `cpvk_cmd.c` came out clean and are trimmed:

`vkEndCommandBuffer`; `CmdBindPipeline`, `CmdBindVertexBuffers2`,
`CmdBindIndexBuffer2`, `CmdPushConstants2`, `CmdSetViewportWithCount`,
`CmdSetScissorWithCount`; `CmdDraw`, `CmdDrawIndexed`, `CmdDrawIndirect`,
`CmdDrawIndexedIndirect`, `CmdDispatchBase`; `CmdBeginRendering`,
`CmdEndRendering`; `CmdFillBuffer`, `CmdCopyBuffer2`, `CmdCopyImage2`,
`CmdCopyBufferToImage2`, `CmdCopyImageToBuffer2`, `CmdBlitImage2`,
`CmdResolveImage2`; `CmdClearColorImage`, `CmdClearDepthStencilImage`,
`CmdClearAttachments`; the six event entry points and the two pipeline
barriers; `CmdResetQueryPool`, `CmdWriteTimestamp2`, `CmdEndQuery`,
`CmdCopyQueryPoolResults`.

Their whole reachable set is host work: `cpvk_op_alloc` (realloc),
`cpvk_scope_append`, `cpvk_record_draw_cmd`, `cpvk_record_copy`,
`cpvk_record_event`, `cpvk_record_query`, `cpvk_plan_batches`,
`cpvk_cmd_retain_*`, the format helpers and `util_format_*`.

**Left alone, each because a CUDA call is reachable:**

- `cpvk_BeginCommandBuffer` — resets and can free device memory.
- `cpvk_CmdBindDescriptorSets2` — `cpvk_arena_append` grows the descriptor
  arena with `cuMemAlloc`/`cuMemFree` when it overflows, which is why that
  function carries a `cp_ctx_check` of its own.
- `cpvk_CmdUpdateBuffer` — copies the inline data to the device at record time.
- `cpvk_CmdExecuteCommands` and `cpvk_execute_dispatch` — these execute.

Everything outside `cpvk_cmd.c` is untouched, as are the object-lifetime entry
points in that file (`CreateDescriptorPool`, `AllocateDescriptorSets`,
`CreateEvent`, `CreateQueryPool`, `GetQueryPoolResults` and friends): several
of them are CUDA-free too, but none of them is on the per-draw path, so they
carry the scope and the risk of a later CUDA call for free.

## What keeps this honest

`CUDAVK_CTX_CHECK` is the whole-call-graph test and it still applies: its
chokepoints are `cp_launch`, `cpvk_arena_append` and the allocators, so a CUDA
call that appears under a trimmed entry point in future names itself instead of
silently running in the application's context. Run the suite with
`CUDAVK_CTX_CHECK=1` after touching a record path.

**Not measured here.** The change is host-side and its predicted size is lead
G's 0.07 ms/frame; the A/B against `CUDAVK_NO_CTX_SCOPE_TRIM=1` belongs to
whoever owns the measurement session.
