project('nvcuda', ['c'], version: 'v0.2', license: 'mixed')

include_path = include_directories('./include')

c_compiler = meson.get_compiler('c')
is_mingw = c_compiler.get_id() == 'gcc' and c_compiler.has_define('__MINGW32__')

lib_dl = is_mingw ? dependency('', required: false) : c_compiler.find_library('dl')

winebuild = find_program('winebuild', required: false)
winegcc   = find_program('winegcc', required: false)
wrc       = find_program('wrc', required: false)

if not is_mingw
  subdir('dlls')
else
  subdir('tests')
endif
