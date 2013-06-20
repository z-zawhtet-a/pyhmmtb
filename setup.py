'''
Created on 20.06.2013

@author: christian
'''

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy as np
from sphinx.setup_command import BuildDoc

def iterPackages(dir, exclude=['.svn']):
    yield dir
    for dirpath, dirnames, filenames in os.walk(dir):
        # walk will not recurse into directories that are removed from dirnames
        for e in exclude:
            try:  
                dirnames.remove(e)
            except:
                pass 
        for dirname in dirnames:
            yield (dirpath+'/'+dirname).replace('/', '.')

setup(name='pyhmmtb',
      version='0.1',
      description='A port of Kevin Murphys Hidden Markov Model Toolbox to Python.',
      author='Christian Vollmer',
      author_email='christian.vollmer@gmx.net',
      packages=list(iterPackages(dir='pyhmmtb', exclude=['.svn'])),
      cmdclass = {'build_ext': build_ext,
                  'build_sphinx': BuildDoc},
      include_dirs = [np.get_include()]
     )