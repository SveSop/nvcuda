nvcuda_res = custom_target(
  'nvcuda.res',
  output: 'nvcuda.res',
  input: 'nvcuda.rc',
  command: [
    wrc,
    '-I', meson.project_source_root() + '/include',
    '-o', '@OUTPUT@',
    '@INPUT@',
  ],
  install: false,
)

nvcuda_def = custom_target(
  'nvcuda.def',
  output: 'nvcuda.def',
  input: '../nvcuda/nvcuda.spec',
  command: [winebuild, target_arch, '-w', '--def', '-o', '@OUTPUT@', '-E', '@INPUT@'],
  install: false,
)

custom_target(
  'nvcuda.dll',
  output: 'nvcuda.dll',
  input: [nvcuda_def, nvcuda_res],
  command: [
    winebuild,
    target_arch,
    '--dll',
    '--fake-module',
    '-o', '@OUTPUT@',
    '-E', '@INPUT@',
  ],
  install: true,
  install_dir: install_dir_windows,
)
