cc = meson.get_compiler('c')
libm = cc.find_library('m', required: true)

walker_lib = shared_library(
  'walker',
  'walker.c',
  install: false,
  build_by_default: true,
  dependencies: [libm]
)

copy_walker = custom_target(
  'copy_walker_so',
  output: 'walker.so',
  command: ['cp', walker_lib.full_path(), meson.current_source_dir()],
  depends: walker_lib,
  build_by_default: true
)
