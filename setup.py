#!/usr/bin/env python
import os
import pkgutil
import sys
from setuptools import setup, find_packages, Extension, Command
from setuptools.command.test import test as TestCommand
from subprocess import check_call, CalledProcessError

try:
    from distutils.config import ConfigParser
except ImportError:
    from configparser import ConfigParser

conf = ConfigParser()
conf.read(['setup.cfg'])

# Get some config values
metadata = dict(conf.items('metadata'))
PACKAGENAME = metadata.get('package_name', 'packagename')
DESCRIPTION = metadata.get('description', '')
AUTHOR = metadata.get('author', 'STScI')
AUTHOR_EMAIL = metadata.get('author_email', 'help@stsci.edu')
URL = metadata.get('url', 'https://www.stsci.edu/')
LICENSE = metadata.get('license', 'BSD')


if not pkgutil.find_loader('relic'):
    relic_local = os.path.exists('relic')
    relic_submodule = (relic_local and
                       os.path.exists('.gitmodules') and
                       not os.listdir('relic'))
    try:
        if relic_submodule:
            check_call(['git', 'submodule', 'update', '--init', '--recursive'])
        elif not relic_local:
            check_call(['git', 'clone', 'https://github.com/spacetelescope/relic.git'])

        sys.path.insert(1, 'relic')
    except CalledProcessError as e:
        print(e)
        exit(1)

import relic.release

version = relic.release.get_info()
relic.release.write_template(version, PACKAGENAME)


# allows you to build sphinx docs from the pacakge
# main directory with python setup.py build_sphinx

try:
    from sphinx.cmd.build import build_main
    from sphinx.setup_command import BuildDoc

    class BuildSphinx(BuildDoc):
        """Build Sphinx documentation after compiling C source files"""

        description = 'Build Sphinx documentation'

        def initialize_options(self):
            BuildDoc.initialize_options(self)

        def finalize_options(self):
            BuildDoc.finalize_options(self)

        def run(self):
            build_cmd = self.reinitialize_command('build_ext')
            build_cmd.inplace = 1
            self.run_command('build_ext')
            build_main(['-b', 'html', './docs', './docs/_build/html'])

except ImportError:
    class BuildSphinx(Command):
        user_options = []

        def initialize_options(self):
            pass

        def finalize_options(self):
            pass

        def run(self):
            # print('!\n! Sphinx is not installed!\n!', file=sys.stderr)
            raise RuntimeError('!\n! Sphinx is not installed!\n!')
            exit(1)

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['pysiaf/tests']
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


setup(
    name=PACKAGENAME,
    version=version.short,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires=[
        'astropy>=1.2',
        'numpy>=1.9',
        'numpydoc',
        'matplotlib>=1.4.3',
        'lxml>=3.6.4',
        'scipy>=0.17',
        'openpyxl>=2.4'
    ],
    tests_require=['pytest'],
    packages=find_packages(),
    package_data={PACKAGENAME: ['prd_data/HST/*/*.dat',
                                'prd_data/JWST/*/*/*/*.xlsx',
                                'prd_data/JWST/*/*/*/*.xml',
                                'pre_delivery_data/*/*.*',
                                'source_data/*/*.txt',
                                'source_data/*.txt',
                                'tests/test_data/*/*/*/*.fits',
                                'tests/test_data/*/*/*/*.txt',
                                ]},
    cmdclass={
        'test': PyTest,
        'build_sphinx': BuildSphinx
    },)
