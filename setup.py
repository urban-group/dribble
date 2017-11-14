from setuptools import setup, find_packages
from codecs import open
import os

__author__ = "Alexander Urban"
__email__ = "aurban@atomistic.net"

here = os.path.abspath(os.path.dirname(__file__))
package_name = 'dribble'
package_description = 'Lattice percolation simulations'

# Get the long description from the README file
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as fp:
    long_description = fp.read()

# Get version number from the VERSION file
with open(os.path.join(here, package_name, 'VERSION')) as fp:
    version = fp.read().strip()

setup(
    name=package_name,
    version=version,
    description=package_description,
    long_description=long_description,
    # url='dribble.atomistic.net',
    author=__author__,
    author_email=__email__,
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
    keywords=['percolation', 'materials science', 'monte carlo'],
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy', 'scipy', 'pymatgen'],
    # package_data={
    #     'sample': [''],
    # },
    entry_points={
        'console_scripts': [
            'dribble=dribble.scripts.dribble:main',
        ],
    },
    # scripts=glob.glob(os.path.join("scripts", "*.py"))
)
