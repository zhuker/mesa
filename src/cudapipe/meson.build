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
)

libcudapipe_vk = static_library(
  'cudapipe_vk',
  [libcudapipe_vk_files, cpvk_entrypoints],
  include_directories : [inc_include, inc_src, inc_util],
  dependencies : [idep_vulkan_runtime, idep_vulkan_util, idep_mesautil,
                  dep_cuda_vk],
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
    '--api-version', '1.0', '--xml', '@INPUT1@',
    '--sizeof-pointer', sizeof_pointer,
    '--icd-lib-path', meson.current_build_dir(),
    '--icd-filename', cpvk_icd_file_name,
    '--out', '@OUTPUT@',
  ],
  build_by_default : true,
)
