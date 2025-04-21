project('walker', 'c')

cc = meson.get_compiler('c')
math_dep = cc.find_library('m', required : false)

add_project_arguments(['-Wall', '-fPIC', '-O2', '-fvisibility=hidden'], language : 'c')

sources = files('walker/walker2d.c', 'walker/walker3d.c')

library(meson.project_name(), sources, dependencies : math_dep)
