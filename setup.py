import os, sys
from glob import glob
# setuptools needs to come before numpy.distutils to get install_requires
#import setuptools 
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

VERSION = "0.27-dev"
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
    'description': "Compute 1D and 2D PMF",
    'long_description': """A package to compute 1D and 2D potential of mean force profiles.
    See README for further details
    """,
   
    }

ext_modules = cythonize( [Extension(
    "cwham",["src/ext/cwham.pyx", "src/ext/helperwham.c"],
    include_dirs = [numpy.get_include()], 
    language="c",
    extra_compile_args=['-fopenmp','-O3'],
    extra_link_args = ["-lgomp"],

    ) ] )

scripts =  [e for e in glob('scripts/*.py') if not e.endswith('__.py')]



setup(packages=["pmfcalculator","pmfcalculator.pmf1d","pmfcalculator.pmf2d","pmfcalculator.tools","pmfcalculator.pmfNd"],
    package_dir = {'pmfcalculator':'src'},
    ext_package = "pmfcalculator",
    ext_modules = ext_modules,
    scripts = scripts,
    **metadata
)

