from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='word2doc',
    version='0.0.1',
    description='Bachelor thesis',
    long_description=readme,
    author='Julian Brendl',
    author_email='julianbrendl@gmail.com',
    url='https://github.com/jundl77/word2doc',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
