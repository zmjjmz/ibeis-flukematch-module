#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function, unicode_literals
from setuptools import setup
try:
    from utool import util_setup
except ImportError:
    print('ERROR: setup requires utool')
    raise

INSTALL_REQUIRES = [
    #'cython >= 0.21.1',
    #'numpy >= 1.9.0',
    #'scipy >= 0.16.0',
]

CLUTTER_PATTERNS = [
    # Patterns removed by python setup.py clean
]

if __name__ == '__main__':
    kwargs = util_setup.setuptools_setup(
        setup_fpath=__file__,
        name='ibeis_flukematch',
        packages=util_setup.find_packages(),
        version=util_setup.parse_package_for_version('ibeis_flukematch'),
        license=util_setup.read_license('LICENSE'),
        long_description=util_setup.parse_readme('README.md'),
        ext_modules=util_setup.find_ext_modules(),
        cmdclass=util_setup.get_cmdclass(),
        #description='description of module',
        #url='https://github.com/<username>/ibeis-flukematch-module.git',
        #author='<author>',
        #author_email='<author_email>',
        keywords='',
        install_requires=INSTALL_REQUIRES,
        clutter_patterns=CLUTTER_PATTERNS,
        #package_data={'build': ut.get_dynamic_lib_globstrs()},
        #build_command=lambda: ut.std_build_command(dirname(__file__)),
        classifiers=[],
    )
    setup(**kwargs)
