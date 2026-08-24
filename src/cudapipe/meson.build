# The native Vulkan target: same CUDA backend, Mesa's common Vulkan runtime in
# front of it instead of lavapipe and Gallium. See CUDAPIPE_VK_NATIVE.md.
#
# It is built beside the Gallium-hosted driver, not instead of it: every step
# of the port is gated by rendering the same frames through both and diffing
# them, which is only possible while both exist.

# The CUDA driver API: headers from the toolkit if pkg-config knows it, and
# libcuda itself always, which the toolkit's own .pc file does not carry.
dep_cuda_vk = declare_dependency(
  dependencies : [
    dependency('cuda', required : false),
    meson.get_compiler('c').find_library('cuda', required : true),
  ],
)

cpvk_entrypoints = custom_target(
  'cpvk_entrypoints',
  input : [vk_entrypoints_gen, vk_api_xml],
  output : ['cpvk_entrypoints.h', 'cpvk_entrypoints.c'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@', '--proto', '--weak',
    '--out-h', '@OUTPUT0@', '--out-c', '@OUTPUT1@', '--prefix', 'cpvk',
    '--beta', with_vulkan_beta.to_string(),
  ],
  depend_files : vk_entrypoints_gen_depend_files,
)

libcudapipe_vk_files = files(
  'cpvk_device.c',
  'cpvk_device_memory.c',
  'cpvk_pipeline.c',
  'cpvk_cmd.c',
  'cpvk_sync.c',
  'cpvk_image.c',
  'cpvk_texture_cache.c',
)

# The renderer, the flag registry, the kernel loader and the NIR-to-PTX
# backend. These were reached across the tree while the Gallium driver was
# being taken apart; they live here now, and src/gallium/drivers/cudapipe is
# back to exactly what it was before this branch started. The Gallium driver
# keeps its own copies of the four files it still needs until it is deleted,
# which is a duplication with a fixed end date and no editing on the other
# side of it.
libcudapipe_vk_shared = files(
  'cp_debug.c',
  'cp_smallop_tele.c',
  'cp_kernels.c',
  'cp_renderer.c',
  'nir_to_ptx/cp_nir_to_llvm.c',
)

# The inline fragment interpolator is LLVM bitcode built by the clang whose
# major version matches Mesa's LLVM dependency. It is embedded at build time;
# the driver never runs a compiler subprocess.
cpvk_llvm_major = dep_llvm.version().split('.')[0]
cpvk_fs_inline_inc = []
cpvk_fs_inline_args = []
cpvk_fs_inline_deps = []
if dep_llvm.version().version_compare('>=18.0.0')
  cpvk_clang = find_program('clang-' + cpvk_llvm_major, required : false)
  if cpvk_clang.found()
    if cpvk_clang.version().split('.')[0] != cpvk_llvm_major
      error('cudapipe inline FS needs clang ' + cpvk_llvm_major +
            ' to match LLVM ' + dep_llvm.version())
    endif
    cpvk_fs_inline_deps += dependency(
      'llvm',
      method : host_machine.system() == 'windows' ? 'auto' : 'config-tool',
      version : '=' + dep_llvm.version(),
      modules : ['bitreader', 'linker', 'passes', 'ipo', 'nvptx'],
      required : true,
      static : not _shared_llvm,
      include_type : 'system',
    )
    cpvk_fs_inline_bc = custom_target(
      'cpvk_fs_inline.bc',
      input : ['kernels/cp_fs_inline.c', 'kernels/cp_fs_interp.h',
               'kernels/cp_rast_types.h'],
      output : 'cp_fs_inline.bc',
      command : [cpvk_clang,
        '-target', 'nvptx64-nvidia-cuda', '-march=sm_86', '--cuda-feature=+ptx75',
        '-O1', '-ffreestanding', '-nostdlib', '-ffp-contract=off',
        '-emit-llvm', '-c', '@INPUT0@',
        '-I' + meson.current_source_dir() + '/kernels', '-o', '@OUTPUT@'],
    )
    cpvk_fs_inline_inc = custom_target(
      'cpvk_fs_inline.bc.inc',
      input : cpvk_fs_inline_bc,
      output : 'cp_fs_inline.bc.inc',
      command : [prog_python, files('kernels/bin_to_inc.py'), '@INPUT@', '@OUTPUT@'],
    )
    cpvk_fs_inline_args += '-DCP_HAVE_FS_INLINE_BC'
  else
    warning('cudapipe: matching clang-' + cpvk_llvm_major +
            ' not found; building with fused/classic fragment interpolation')
  endif
else
  warning('cudapipe: LLVM ' + dep_llvm.version() +
          ' is too old for linked inline fragment interpolation; using fused/classic')
endif

# NVRTC compiles the kernels from source at runtime, so stringify each .cu
# into a C string the driver can hand it.
cpvk_kernel_incs = []
foreach kernel : ['cp_clear.cu', 'cp_rasterize.cu', 'cp_sampler.cu', 'cp_math.cu', 'cp_fs.cu',
                  'cp_vertex_fetch.cu', 'cp_rast_types.h', 'cp_fs_interp.h']
  cpvk_kernel_incs += custom_target(
    'cpvk_' + kernel + '.inc',
    input : ['kernels/' + kernel],
    output : [kernel + '.inc'],
    command : [prog_python, files('kernels/cu_to_inc.py'), '@INPUT@', '@OUTPUT@'],
  )
endforeach

# The NVTX timeline ranges, exactly as the Gallium target enables them.
#
# cp_nvtx.h compiles every push and pop to nothing without this, so the native
# build emitted no ranges at all and cp_prof_nvtx.py reported "no NVTX_EVENTS
# table" -- while the same renderer sources, built for the Gallium target,
# named every stage. That is the driver's own answer to every shader being a
# CUDA kernel called `main`, and this build had quietly lost it.
cpvk_nvtx_args = []
if meson.get_compiler('c').has_header('nvtx3/nvToolsExt.h')
  cpvk_nvtx_args += '-DCP_HAVE_NVTX'
endif

libcudapipe_vk = static_library(
  'cudapipe_vk',
  [libcudapipe_vk_files, libcudapipe_vk_shared, cpvk_entrypoints,
   cpvk_kernel_incs, cpvk_fs_inline_inc],
  c_args : [cpvk_nvtx_args, cpvk_fs_inline_args],
  include_directories : [inc_include, inc_src, inc_util,
                         include_directories('.'),
                         include_directories('nir_to_ptx'),
                         include_directories('../compiler/nir')],
  dependencies : [idep_vulkan_runtime, idep_vulkan_util, idep_vulkan_wsi,
                  idep_mesautil, idep_nir, dep_llvm, cpvk_fs_inline_deps,
                  dep_cuda_vk, dep_nvrtc, dep_nvjitlink],
  gnu_symbol_visibility : 'hidden',
)

libvulkan_cudapipe_native = shared_library(
  'vulkan_cudapipe_native',
  ['cpvk_target.c'],
  vs_module_defs : vulkan_api_def,
  include_directories : [inc_include, inc_src, inc_util],
  link_whole : [libcudapipe_vk],
  # The runtime and util deps come through libcudapipe_vk; naming them again
  # here makes meson list the same archives twice under --whole-archive.
  dependencies : [dep_cuda_vk],
  gnu_symbol_visibility : 'hidden',
  link_args : [ld_args_bsymbolic, ld_args_gc_sections],
  name_prefix : [],
  install : true,
)

cpvk_icd_file_name = 'libvulkan_cudapipe_native.so'

cpvk_devenv_icd = custom_target(
  'cpvk_devenv_icd',
  input : [vk_icd_gen, vk_api_xml],
  output : 'cudapipe_native_devenv_icd.@0@.json'.format(host_machine.cpu()),
  command : [
    prog_python, '@INPUT0@',
    # Match the driver's implemented Vulkan 1.1 contract. The loader clamps
    # application dispatch to this value, so leaving the old 1.0 here hid core
    # 1.1 entrypoints even though the physical device reported them. Do not
    # raise this to 1.3: that would promise mandatory features not implemented.
    '--api-version', '1.1', '--xml', '@INPUT1@',
    '--sizeof-pointer', sizeof_pointer,
    '--icd-lib-path', meson.current_build_dir(),
    '--icd-filename', cpvk_icd_file_name,
    '--out', '@OUTPUT@',
  ],
  build_by_default : true,
)


# Focused external-API regression suite. These link the Vulkan loader rather
# than driver internals and select the generated development ICD, so they test
# the same dispatch and object-lifetime boundary as applications do.
dep_vulkan_loader = dependency('vulkan', required : false)
if dep_vulkan_loader.found()
  cpvk_test_env = environment()
  cpvk_test_env.set('VK_DRIVER_FILES', cpvk_devenv_icd.full_path())
  cpvk_test_env.set('VK_ICD_FILENAMES', cpvk_devenv_icd.full_path())
  cpvk_test_names = [
    'cpvk_batch',
    'cpvk_batchbig',
    'cpvk_batchblend',
    'cpvk_batchclip',
    'cpvk_batchpush',
    'cpvk_batchtex',
    'cpvk_bccopy',
    'cpvk_compute_pipeline_lifetime',
    'cpvk_cubebig',
    'cpvk_cubebig_pbrsamp',
    'cpvk_cubedir',
    'cpvk_cubelod',
    'cpvk_cubelodf16',
    'cpvk_cubesph',
    'cpvk_cubesph_nearest',
    'cpvk_depth_attachment',
    'cpvk_desc_array',
    'cpvk_descriptor_pool',
    'cpvk_descriptor_semantics',
    'cpvk_draw',
    'cpvk_event_state',
    'cpvk_fixed_edge',
    'cpvk_lod',
    'cpvk_lodblit',
    'cpvk_lodimp',
    'cpvk_mesh',
    'cpvk_multidevice',
    'cpvk_pipeline_lifetime',
    'cpvk_query_async',
    'cpvk_query_lifetime',
    'cpvk_rtt',
    'cpvk_sampler_array',
    'cpvk_sampler_two_bindings',
    'cpvk_texture_cache_descriptors',
    'cpvk_texture_cache_ops',
    'cpvk_texture_cache_no_replay',
    'cpvk_secondary_descriptor',
    'cpvk_push_stages',
    'cpvk_smoke',
    'cpvk_ssbo_length',
    'cpvk_static_viewport',
    'cpvk_sync_state',
    'cpvk_tex',
    'cpvk_tex3d',
    'cpvk_tri',
    'cpvk_ubo',
    'cpvk_wsi',
  ]
  foreach test_name : cpvk_test_names
    test_exe = executable(test_name, files('tests/' + test_name + '.c'),
                          dependencies : dep_vulkan_loader,
                          c_args : ['-Wno-error=missing-prototypes'],
                          build_by_default : false)
    test(test_name, test_exe,
         suite : 'cudapipe-native',
         env : cpvk_test_env,
         timeout : 120,
         is_parallel : false,
         depends : [libvulkan_cudapipe_native, cpvk_devenv_icd])
    if test_name == 'cpvk_sampler_two_bindings'
      cpvk_texture_cache_test_exe = test_exe
    endif
    if test_name == 'cpvk_texture_cache_no_replay'
      cpvk_texture_cache_no_replay_exe = test_exe
    endif
  endforeach
  cpvk_texture_cache_formats_exe = executable(
    'cpvk_texture_cache_formats',
    files('tests/cpvk_texture_cache_formats.c'),
    dependencies : dep_vulkan_loader,
    c_args : ['-Wno-error=missing-prototypes'],
    build_by_default : false)
  test('cp_launch_audit', prog_python,
       args : [files('tests/cp_launch_audit.py')],
       suite : 'cudapipe-native')
  test('cpvk_texture_cache_formats_gate', prog_python,
       args : [files('tests/cpvk_texture_cache_formats_gate.py'),
               cpvk_texture_cache_formats_exe],
       suite : 'cudapipe-native', env : cpvk_test_env, timeout : 120,
       is_parallel : false,
       depends : [libvulkan_cudapipe_native, cpvk_devenv_icd])
  test('cpvk_texture_cache_gate', prog_python,
       args : [files('tests/cpvk_texture_cache_gate.py'),
               cpvk_texture_cache_test_exe],
       suite : 'cudapipe-native', env : cpvk_test_env, timeout : 120,
       is_parallel : false,
       depends : [libvulkan_cudapipe_native, cpvk_devenv_icd])
  test('cpvk_texture_cache_fault_gate', prog_python,
       args : [files('tests/cpvk_texture_cache_fault_gate.py'),
               cpvk_texture_cache_test_exe],
       suite : 'cudapipe-native', env : cpvk_test_env, timeout : 120,
       is_parallel : false,
       depends : [libvulkan_cudapipe_native, cpvk_devenv_icd])
  test('cpvk_texture_cache_purge_gate', prog_python,
       args : [files('tests/cpvk_texture_cache_purge_gate.py'),
               cpvk_texture_cache_test_exe],
       suite : 'cudapipe-native', env : cpvk_test_env, timeout : 120,
       is_parallel : false,
       depends : [libvulkan_cudapipe_native, cpvk_devenv_icd])
  test('cpvk_texture_cache_oom_gate', prog_python,
       args : [files('tests/cpvk_texture_cache_oom_gate.py'),
               cpvk_texture_cache_test_exe],
       suite : 'cudapipe-native', env : cpvk_test_env, timeout : 120,
       is_parallel : false,
       depends : [libvulkan_cudapipe_native, cpvk_devenv_icd])
  test('cpvk_texture_cache_array_oom_gate', prog_python,
       args : [files('tests/cpvk_texture_cache_array_oom_gate.py'),
               cpvk_texture_cache_test_exe],
       suite : 'cudapipe-native', env : cpvk_test_env, timeout : 120,
       is_parallel : false,
       depends : [libvulkan_cudapipe_native, cpvk_devenv_icd])
  test('cpvk_texture_cache_object_oom_gate', prog_python,
       args : [files('tests/cpvk_texture_cache_object_oom_gate.py'),
               cpvk_texture_cache_test_exe],
       suite : 'cudapipe-native', env : cpvk_test_env, timeout : 120,
       is_parallel : false,
       depends : [libvulkan_cudapipe_native, cpvk_devenv_icd])
  test('cpvk_texture_cache_conversion_fault_gate', prog_python,
       args : [files('tests/cpvk_texture_cache_conversion_fault_gate.py'),
               cpvk_texture_cache_formats_exe],
       suite : 'cudapipe-native', env : cpvk_test_env, timeout : 120,
       is_parallel : false,
       depends : [libvulkan_cudapipe_native, cpvk_devenv_icd])
  test('cpvk_texture_cache_no_replay_fault',
       cpvk_texture_cache_no_replay_exe,
       args : ['--fault'],
       suite : 'cudapipe-native', env : cpvk_test_env, timeout : 120,
       is_parallel : false,
       depends : [libvulkan_cudapipe_native, cpvk_devenv_icd])
  test('cpvk_texture_cache_fs_arg_fault_gate', prog_python,
       args : [files('tests/cpvk_texture_cache_fs_arg_fault_gate.py'),
               cpvk_texture_cache_no_replay_exe],
       suite : 'cudapipe-native', env : cpvk_test_env, timeout : 120,
       is_parallel : false,
       depends : [libvulkan_cudapipe_native, cpvk_devenv_icd])
  test('cpvk_texture_cache_create_fault_gate', prog_python,
       args : [files('tests/cpvk_texture_cache_create_fault_gate.py'),
               cpvk_texture_cache_no_replay_exe,
               cpvk_texture_cache_formats_exe],
       suite : 'cudapipe-native', env : cpvk_test_env, timeout : 120,
       is_parallel : false,
       depends : [libvulkan_cudapipe_native, cpvk_devenv_icd])
endif
