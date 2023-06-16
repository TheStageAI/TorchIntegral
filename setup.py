import os
import platform
import numpy as np
from setuptools import setup
from setuptools.command.build_ext import build_ext
from distutils.extension import Extension
from Cython.Build import cythonize


#--------------------------------------------------------------------------------------------
# Handling compile args for Cython
#--------------------------------------------------------------------------------------------
if platform.system() == 'Darwin':
    compile_opts = [ 
        '-std=c++11', '-Ofast', #'-fopenmp',
        '-mmacosx-version-min={:}'.format( platform.mac_ver()[0] ),
    ]

elif platform.system() == 'Linux':
    compile_opts = [ 
        '-std=c++11', '-Ofast', #'-fopenmp'
    ]

else:
    raise EnvironmentError(
        'Not supported platform: {plat}'.format(plat=platform.system()) 
    )


#--------------------------------------------------------------------------------------------
# C++/Cython extesnions and packages
#--------------------------------------------------------------------------------------------
tsp_ext = Extension( 
    "torch_integral.tsp_solver.solver",
    sources=["torch_integral/tsp_solver/solver.pyx"],
    extra_compile_args=compile_opts,
    extra_link_args=compile_opts,
    language = "c++",
    include_dirs=[np.get_include()],
)
ext_modules = [tsp_ext]
packages = ['torch_integral', 'torch_integral.tsp_solver']

#--------------------------------------------------------------------------------------------
# Package setup
#--------------------------------------------------------------------------------------------
setup( 
    name='TorchIntegral',
    ext_modules = cythonize(ext_modules),
    version='0.0.0.0',
    author='Azim Kurbanov, Solodskikh Kirill',
    author_email='hello@thestage.ai',
    maintainer='Intuition',
    maintainer_email='hello@thestage.ai',
    install_requires=['cython'],
    description='Official Integral Neural Networks in PyTorch.',
    url='https://inn.thestage.ai',
    zip_safe=False,
    packages=packages,
    license='Apache License 2.0', 
    long_description='Bla Bla',
    classifiers=['Programming Language :: Python :: 3' ] 
)
