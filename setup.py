# filepath: /Users/arnaou/projects/dtu_chemeng/camd_iss/setup.py
from setuptools import setup, find_packages

setup(
    name="sgreen",  # Set the package name to 'sgreen'
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],  # Add dependencies here if needed
)