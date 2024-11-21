project('nvcuda', ['c'], version: 'v0.0', license: 'mixed')

include_path = include_directories('./include')

lib_dl      = meson.get_compiler('c').find_library('dl')

winebuild   = find_program('winebuild')
winegcc     = find_program('winegcc')
wrc         = find_program('wrc')

target_arch = target_machine.cpu_family() == 'x86_64' ? '-m64' : '-m32'

subdir('dlls')
