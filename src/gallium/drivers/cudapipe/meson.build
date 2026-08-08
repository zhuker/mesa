files_cudapipe = files(
  'cp_screen.c',
  'cp_context.c',
  'cp_resource.c',
  'cp_kernels.c',
  'nir_to_ptx/cp_nir_to_llvm.c',
)

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

libcudapipe = static_library(
  'cudapipe',
  files_cudapipe,
  include_directories : [inc_include, inc_src, inc_util, inc_gallium, inc_gallium_aux, inc_cudapipe],
  dependencies : [idep_nir, idep_mesautil, dep_cuda, dep_nvrtc, dep_nvjitlink, dep_llvm],
  gnu_symbol_visibility : 'hidden',
)

driver_cudapipe = declare_dependency(
  compile_args : '-DGALLIUM_CUDAPIPE',
  link_with : [libcudapipe],
  link_args : ['-lcuda', '-lnvrtc', '-lnvJitLink'],
  include_directories : [inc_cudapipe],
)
