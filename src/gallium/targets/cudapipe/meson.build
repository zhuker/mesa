libvulkan_cudapipe = shared_library(
  'vulkan_cudapipe',
  ['cudapipe_target.c'],
  vs_module_defs : vulkan_api_def,
  include_directories : [inc_src, inc_util, inc_include, inc_gallium, inc_gallium_aux, inc_gallium_winsys, inc_gallium_drivers],
  link_whole : [liblavapipe_st],
  link_with : [libpipe_loader_static, libgallium, libwsw, libswdri, libws_null, libswkmsdri],
  gnu_symbol_visibility : 'hidden',
  link_args : [ld_args_bsymbolic, ld_args_gc_sections],
  dependencies : [driver_cudapipe, driver_llvmpipe, idep_mesautil],
  name_prefix : host_machine.system() == 'windows' ? '' : [],
  install : true,
)

cudapipe_icd_file_name = libname_prefix + 'vulkan_cudapipe.' + libname_suffix

cudapipe_icd = custom_target(
  'cudapipe_icd',
  input : [vk_icd_gen, vk_api_xml],
  output : 'cudapipe_icd.@0@.json'.format(host_machine.cpu()),
  command : [
    prog_python, '@INPUT0@',
    '--api-version', '1.4', '--xml', '@INPUT1@',
    '--sizeof-pointer', sizeof_pointer,
    '--icd-lib-path', vulkan_icd_lib_path,
    '--icd-filename', cudapipe_icd_file_name,
    '--out', '@OUTPUT@',
  ],
  build_by_default : true,
  install_dir : with_vulkan_icd_dir,
  install_tag : 'runtime',
  install : true,
)

# Same manifest but pointing at the build tree, so the driver can be used
# without installing it.
cudapipe_devenv_icd = custom_target(
  'cudapipe_devenv_icd',
  input : [vk_icd_gen, vk_api_xml],
  output : 'cudapipe_devenv_icd.@0@.json'.format(host_machine.cpu()),
  command : [
    prog_python, '@INPUT0@',
    '--api-version', '1.4', '--xml', '@INPUT1@',
    '--sizeof-pointer', sizeof_pointer,
    '--icd-lib-path', meson.current_build_dir(),
    '--icd-filename', cudapipe_icd_file_name,
    '--out', '@OUTPUT@',
  ],
  build_by_default : true,
)
