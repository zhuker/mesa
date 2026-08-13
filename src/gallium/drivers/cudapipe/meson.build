files_cudapipe = files(
  'cp_screen.c',
  'cp_context.c',
  'cp_resource.c',
  'cp_kernels.c',
  'nir_to_ptx/cp_nir_to_llvm.c',
)

# NVRTC compiles the kernels from source at runtime, so stringify each .cu
# (and the header they share) into a C string literal at build time.
cu_to_inc = files('kernels/cu_to_inc.py')

cudapipe_kernel_incs = []
foreach kernel : ['cp_clear.cu', 'cp_rasterize.cu', 'cp_sampler.cu', 'cp_fs.cu',
                  'cp_vertex_fetch.cu', 'cp_rast_types.h']
  cudapipe_kernel_incs += custom_target(
    kernel + '.inc',
    input : files('kernels/' + kernel),
    output : kernel + '.inc',
    command : [prog_python, cu_to_inc, '@INPUT@', '@OUTPUT@'],
  )
endforeach

inc_cudapipe = include_directories('.')

dep_cuda = dependency('cuda', required : false)
if not dep_cuda.found()
  dep_cuda = declare_dependency(
    dependencies : [
      meson.get_compiler('c').find_library('cuda', required : true),
    ],
  )
endif

dep_nvrtc = meson.get_compiler('c').find_library('nvrtc', required : false)
dep_nvjitlink = meson.get_compiler('c').find_library('nvJitLink', required : false)

# NVTX is header-only in its v3 form — the headers load the injection library
# themselves when a profiler is attached, so there is nothing to link and
# nothing to pay for when there is no profiler. Optional so that a CUDA install
# without the headers still builds; the ranges compile out.
cp_nvtx_args = []
if meson.get_compiler('c').has_header('nvtx3/nvToolsExt.h')
  cp_nvtx_args += '-DCP_HAVE_NVTX'
endif

libcudapipe = static_library(
  'cudapipe',
  [files_cudapipe, cudapipe_kernel_incs],
  include_directories : [inc_include, inc_src, inc_util, inc_gallium, inc_gallium_aux, inc_cudapipe],
  dependencies : [idep_nir, idep_mesautil, dep_cuda, dep_nvrtc, dep_nvjitlink, dep_llvm],
  c_args : cp_nvtx_args,
  gnu_symbol_visibility : 'hidden',
)

driver_cudapipe = declare_dependency(
  compile_args : '-DGALLIUM_CUDAPIPE',
  link_with : [libcudapipe],
  link_args : ['-lcuda', '-lnvrtc', '-lnvJitLink'],
  include_directories : [inc_cudapipe],
)
