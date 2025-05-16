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

# Handle math library differently based on OS
is_windows = host_machine.system() == 'windows'
dependencies = []

# On non-Windows platforms, find and use libm
if not is_windows
  libm = cc.find_library('m', required: true)
  dependencies += [libm]
endif

# Determine the correct file extension based on OS
lib_extension = is_windows ? 'dll' : 'so'

# Build the library
walker_lib = shared_library(
  'walker',
  'walker.c',
  install: false,
  build_by_default: true,
  dependencies: dependencies
)

# Cross-platform file copy command
copy_walker = custom_target(
  'copy_walker_lib',
  output: 'walker.' + lib_extension,
  input: walker_lib,
  command: [
    find_program('python3', 'python', required: true),
    '-c',
    'import shutil; import os; shutil.copy("@INPUT@", os.path.join("@SOURCE_DIR@", "walker.' + lib_extension + '"))'
  ],
  depends: walker_lib,
  build_by_default: true
)
