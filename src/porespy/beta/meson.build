# cc = meson.get_compiler('c')
# libm = cc.find_library('m', required: true)
# 
# walker_lib = shared_library(
#   'walker',
#   'walker.c',
#   install: false,
#   build_by_default: true,
#   dependencies: [libm]
# )
# 
# copy_walker = custom_target(
#   'copy_walker_so',
#   output: 'walker.so',
#   command: ['cp', walker_lib.full_path(), meson.current_source_dir()],
#   depends: walker_lib,
#   build_by_default: true
# )

cc = meson.get_compiler('c')
libm = cc.find_library('m', required: true)

walker_lib = shared_library(
  'walker',
  'walker.c',
  install: false,
  build_by_default: true,
  dependencies: [libm]
)

lib_filename = walker_lib.full_path().split('/')[-1]

copy_walker = configure_file(
  input: walker_lib,
  output: lib_filename,
  copy: true
)
