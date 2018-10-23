#!/usr/bin/env python

from setuptools import find_packages, setup

import versioneer

install_requires = [
    'numpy<1.15.0',
    'astropy',
    'attrs>=17.2.0',
    'click',
    'drive-casa>=0.7.6',
    'pandas',
    'pytest',
    'six',
    'scipy',
    'tqdm',
]


entry_points=('''
        [console_scripts]
        fastimg_casa_ms_to_npz=fastimgproto.scripts.casa.ms_to_npz:cli
        fastimg_casa_compare_imagers=fastimgproto.scripts.casa.compare_imagers:cli
        fastimg_casa_simulate_vis=fastimgproto.scripts.casa.simulate_vis:cli
        fastimg_extract_lsm=fastimgproto.scripts.extract_lsm:cli
        fastimg_image=fastimgproto.scripts.image:cli
        fastimg_make_config=fastimgproto.scripts.config:cli
        fastimg_reduce=fastimgproto.scripts.reduce:cli
        fastimg_simulate_data=fastimgproto.scripts.simulate_data:cli
        fastimg_sourcefind=fastimgproto.scripts.sourcefind:cli
        ''')


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
    entry_points=entry_points,
)
