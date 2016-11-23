#!/usr/bin/env python

from setuptools import setup, find_packages
import versioneer

install_requires = [
    'astropy',
    'attrs',
    'click',
    'drive-casa>=0.7.6',
    'pandas',
    'pytest',
    'six',
    'scipy',
]

setup(
    name="fastimgproto",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    # description="Utility scripts ",
    author="Tim Staley",
    author_email="github@timstaley.co.uk",
    # url="https://github.com/",
    install_requires=install_requires,
    entry_points='''
        [console_scripts]
        fastimg_casavis_to_npz=fastimgproto.scripts.casavis_to_npz:cli
        fastimg_compare_imagers=fastimgproto.scripts.compare_imagers:cli
        fastimg_extract_lsm=fastimgproto.scripts.extract_lsm:cli
        fastimg_simpipe=fastimgproto.scripts.simpipe:cli
        fastimg_simple_imager=fastimgproto.scripts.simple_imager:cli
        fastimg_simulate_vis_with_casa=fastimgproto.scripts.simulate_vis_with_casa:cli
        fastimg_sourcefind=fastimgproto.scripts.sourcefind:cli
        ''',
)
