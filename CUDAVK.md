# cudavk

A CUDA implementation of Vulkan. It is a Mesa fork, but only one thing in it is
being worked on: `src/cudavk`, which is an ICD built on Mesa's common Vulkan
runtime with a CUDA renderer behind it. There is no Gallium in it.

    meson setup build -Dcudavk=true -Dgallium-drivers= -Dvulkan-drivers=
    ninja -C build
    export VK_DRIVER_FILES=$PWD/build/src/cudavk/cudavk_devenv_icd.x86_64.json

It advertises Vulkan 1.1. On the two GFXR captures this project is measured
against it runs at **15.74 ms/frame** and **5.87 ms/frame**, from 23.07 and 7.61
when the work started.

## Where to read next

| you want to | read |
|---|---|
| build, install and use it from scratch | `docs/cudavk/GETTING_STARTED.md` |
| pick up the performance work | `docs/cudavk/PERF_HANDOFF.md` |
| change the driver | `docs/cudavk/ARCHITECTURE.md` |
| run a measured iteration | `docs/cudavk/WORKFLOW.md` |
| know what is slow and what it costs to fix | `docs/cudavk/PERFORMANCE.md` |
| avoid re-implementing something that failed | `docs/cudavk/DEAD_ENDS.md` |
| know what is unfinished | `docs/cudavk/TODO.md` |
| check correctness | `docs/cudavk/TESTING.md` |
| hand frames to CUDA, or share memory with it | `docs/cudavk/CUDA_INTEROP.md` |
| know what every environment switch does | `src/cudavk/FLAGS.md` (generated) |

`docs/cudavk/history/` keeps the long-form records: the decision log, the
iteration-by-iteration performance record, and the documents from the
Gallium-hosted driver this replaced. `docs/cudavk/notes/` keeps research that
has not been acted on. `docs/cudavk/GALLIUM_RETIREMENT.md` records what the old
driver could do that this one cannot, and how to bring it back.

## Two things that are easy to get wrong

**Environment switches are a registry, not `getenv`.** All 97 live in one array
in `src/cudavk/cp_debug.c`, which is what makes `CUDAVK_HELP=1` and the
generated `FLAGS.md` correct by construction. A new switch goes in that array.
`src/cudavk/tests/cp_debug_doc.py --check` fails if the documentation has
drifted.

**Frame time on these captures is the interval between every *other* submit**,
because they submit twice per frame. Measuring every submit understates the
frame by about 25% and has produced at least one confident wrong answer.
`docs/cudavk/WORKFLOW.md` has the rest of the measurement conventions, and they
exist because each one was learned by getting it wrong.
