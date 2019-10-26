'''A file for setting up the package.'''
import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print(f'This Python is only compatible with Python 3, but you are running '
           'Python {sys.version_info.major}. The installation will likely '
           'fail.')

setup(name='pynn',
      packages=[package for package in find_packages()
                if package.startswith('pynn')],
      description='A Neural Network framework entirely out of python',
      author='Adolfo Gonzalez III',
      url='',
      author_email='',
      keywords="",
      license="",
      )
