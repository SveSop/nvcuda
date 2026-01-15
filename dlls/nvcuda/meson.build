nvcuda_src = [
  'internal.c',
  'nvcuda.c',
  'function_mappings.c',
  'encryption.c',
]

thread_dep = dependency('threads')


shared_library(
  'nvcuda.dll',
  nvcuda_src,
  name_prefix: '',
  dependencies: [thread_dep, lib_dl],
  include_directories: include_path,
  link_depends: ['nvcuda.spec'],
  link_args: [meson.current_source_dir() / 'nvcuda.spec'],
  install: true,
  install_dir: install_dir_unix,
)
