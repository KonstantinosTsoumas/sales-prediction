from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        requirements = [line.strip() for line in f.readlines() if line.strip() != '-e .']
    return requirements

setup(
    name='sales-prediction',
    version='0.0.1',
    author='Konstantinos',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
