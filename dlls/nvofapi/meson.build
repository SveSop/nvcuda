nvofapi_src = [
  'nvofapi.c',
]

thread_dep = dependency('threads')


nvofapi_dll = shared_library('nvofapi64.dll', nvofapi_src,
  name_prefix         : '',
  dependencies        : [ thread_dep, lib_dl ],
  include_directories : include_path,
  objects             : 'nvofapi.spec',
  install             : true)
