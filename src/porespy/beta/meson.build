project('walker', 'c',
  default_options : [
    'warning_level=3',
    'optimization=2',
    'b_lto=false',  # optional, just to be explicit
    'b_pie=true',   # enables -fPIC where needed
    'c_std=c11'     # optional, if you want a specific C standard
  ]
)

cc = meson.get_compiler('c')
math_dep = cc.find_library('m', required : false)

add_project_arguments('-fvisibility=hidden', language : 'c')

sources = files('walker.c')

library(meson.project_name(), sources, dependencies : math_dep)
