py.install_sources(
  '__init__.py',
  'interaction.py',
  'constants.py',
  'hartree_fock.py',
  'susceptibility.py',
  'chemical_potential.py',
  'soc_parameters.py',
  subdir: 'lrpa',
)

subdir('models')

# NumPy/f2py includes
incdir_numpy = run_command(
  py, ['-c', 'import numpy; print(numpy.get_include())'],
  check: true,
).stdout().strip()

incdir_f2py = run_command(
  py, ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true,
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

fortranobject_c = incdir_f2py / 'fortranobject.c'

# Generate f2py wrappers
chi_module_source = custom_target(
  'chi_module_f2py',
  input: 'calculate_chi.f90',
  output: ['chi_modulemodule.c', 'chi_module-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'chi_module',
            '--lower', '--build-dir', '@OUTDIR@'],
)

omp = dependency('openmp', required: false)

py.extension_module(
  'chi_module',
  ['calculate_chi.f90', chi_module_source, fortranobject_c],
  include_directories: inc_np,
  dependencies: [py.dependency(), omp],
  subdir: 'lrpa',
  install: true,
)
