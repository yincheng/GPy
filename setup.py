#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy
from setuptools import setup, Extension
from Cython.Distutils import build_ext

# Version number
version = '0.4.6'

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

compile_flags = ["-march=native"]
ext_modules = [Extension(name="GPy.util.cython.linalg",
                         language="c++",
                         sources=["GPy/util/cython/linalg.pyx"],
                         include_dirs=[numpy.get_include()],
                         extra_compile_args=compile_flags),
              Extension(name="GPy.util.cython.gaussian",
                         language="c++",
                         sources=["GPy/util/cython/gaussian_utils.pyx"],
                         include_dirs=[numpy.get_include()],
                         extra_compile_args=compile_flags),
              Extension(name="GPy.kern.cython.kernels",
                         language="c++",
                         sources=["GPy/kern/cython/kernels.pyx"],
                         extra_link_args=['-fopenmp'],
                         include_dirs=[numpy.get_include()],
                         extra_compile_args=compile_flags+['-fopenmp'],
                         )

  ]

setup(name = 'GPy',
      version=version,
      author=read('AUTHORS.txt'),
      author_email="james.hensman@gmail.com",
      description=("The Gaussian Process Toolbox"),
      license="BSD 3-clause",
      keywords="machine-learning gaussian-processes kernels",
      url="http://sheffieldml.github.com/GPy/",
      packages = ["GPy.models", "GPy.inference.optimization", "GPy.inference", "GPy.inference.latent_function_inference", "GPy.likelihoods", "GPy.mappings", "GPy.examples", "GPy.core.parameterization", "GPy.core", "GPy.testing", "GPy", "GPy.util", "GPy.kern", "GPy.kern._src.psi_comp", "GPy.kern._src", "GPy.plotting.matplot_dep.latent_space_visualizations.controllers", "GPy.plotting.matplot_dep.latent_space_visualizations", "GPy.plotting.matplot_dep", "GPy.plotting"],
      package_dir={'GPy': 'GPy'},
      package_data={'GPy': ['GPy/examples', 'gpy_config.cfg', 'util/data_resources.json']},
      py_modules=['GPy.__init__'],
      long_description=read('README.md'),
      install_requires=['numpy>=1.8', 'scipy>=0.12','matplotlib>=1.2', 'nose'],
      extras_require={
        'docs':['Sphinx', 'ipython'],
      },
      classifiers=[
      "License :: OSI Approved :: BSD License"],
      cmdclass={'build_ext': build_ext, 'inplace': True},
      ext_modules=ext_modules
      #ext_modules =  [Extension(name = 'GPy.kern.lfmUpsilonf2py',
      #          sources = ['GPy/kern/src/lfmUpsilonf2py.f90'])],
      )
