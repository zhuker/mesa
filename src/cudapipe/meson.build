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
  'cp_kernels.c',
  'cp_renderer.c',
  'nir_to_ptx/cp_nir_to_llvm.c',
)

# NVRTC compiles the kernels from source at runtime, so stringify each .cu
# into a C string the driver can hand it.
cpvk_kernel_incs = []
foreach kernel : ['cp_clear.cu', 'cp_rasterize.cu', 'cp_sampler.cu', 'cp_fs.cu',
                  'cp_vertex_fetch.cu', 'cp_rast_types.h']
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
   cpvk_kernel_incs],
  c_args : cpvk_nvtx_args,
  include_directories : [inc_include, inc_src, inc_util,
                         include_directories('.'),
                         include_directories('nir_to_ptx'),
                         include_directories('../compiler/nir')],
  dependencies : [idep_vulkan_runtime, idep_vulkan_util, idep_mesautil,
                  idep_nir, dep_llvm, dep_cuda_vk, dep_nvrtc, dep_nvjitlink],
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
    # 1.0, and measured, not assumed.
    #
    # The loader reads this and rewrites the application's requested
    # apiVersion down to it before calling in, so a manifest saying 1.0 makes
    # every 1.1-and-later core entrypoint dispatch to NULL however high
    # vkEnumerateInstanceVersion and the physical device report. Raising it to
    # 1.3 fixes that -- and costs 21% and 39% of replay time, because
    # gfxrecon-replay then takes its Vulkan 1.3 paths through this driver:
    # 29.34 ms against 24.34 on Crossroads, 109.76 against 79.20 on the older
    # capture, both otherwise identical runs.
    #
    # The objective is no regression, so it stays at 1.0 until those paths are
    # as cheap as the ones they replace. What is behind the manifest is ready:
    # the features are declared and the entry points exist.
    '--api-version', '1.0', '--xml', '@INPUT1@',
    '--sizeof-pointer', sizeof_pointer,
    '--icd-lib-path', meson.current_build_dir(),
    '--icd-filename', cpvk_icd_file_name,
    '--out', '@OUTPUT@',
  ],
  build_by_default : true,
)
