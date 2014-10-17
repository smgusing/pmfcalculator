from glob import glob
from distutils.core import setup

VERSION = "0.3-dev"
#ISRELEASED = False
__author__ = "Gurpreet"
__version__ = VERSION

# metadata for setup()
metadata = {
    'name': "pmfcalculator",
    'version': VERSION,
    'author': __author__,
    'license': 'GPL version 2 or later',
    'author_email': 'togurpreet@gmail.com',
    'url': 'https://sites.google.com/site/togurpreet/Home',
    'platforms': ["Linux"],
    'description': "Compute Potential of Mean Force Profiles",
    'long_description': """A package to compute Potential of mean force profiles.
    """,
   
    }

# ext_modules = cythonize( [Extension(
#     "cwham",["src/ext/cwham.pyx", "src/ext/helperwham.c"],
#     include_dirs = [numpy.get_include()], 
#     language="c",
#     extra_compile_args=['-fopenmp','-O3'],
#     extra_link_args = ["-lgomp"],
# 
#     ) ] )

scripts =  [e for e in glob('scripts/*.py') if not e.endswith('__.py')]



setup(packages=["pmfcalculator","pmfcalculator.pmfNd"],
    package_dir = {'pmfcalculator':'src'},
    ext_package = "pmfcalculator",
#    ext_modules = ext_modules,
    scripts = scripts,
    **metadata
)

