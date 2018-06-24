#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='word2doc',
    version='0.9.0',
    description='Bachelor thesis',
    long_description=readme,
    author='Julian Brendl',
    author_email='julianbrendl@gmail.com',
    url='https://github.com/jundl77/word2doc',
    license=license,
    python_requires='>=3.5',
    packages=find_packages(exclude=('tests', 'docs', 'data')),
    install_requires=reqs.strip().split('\n'),
)
