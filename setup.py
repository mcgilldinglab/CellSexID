#!/usr/bin/env python3
"""
Setup script for CellSexID
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "CellSexID: Single-Cell Sex Identification Tool"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'numpy>=1.19.0',
        'pandas>=1.3.0',
        'scanpy>=1.8.0',
        'scikit-learn>=1.0.0',
        'xgboost>=1.5.0',
        'matplotlib>=3.3.0',
        'anndata>=0.8.0'
    ]

setup(
    name="cellsexid",
    version="1.0.0",
    author="Your Name",  # Replace with your name
    author_email="your.email@example.com",  # Replace with your email
    description="Single-Cell Sex Identification Tool with Machine Learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/CellSexID",  # Replace with your GitHub URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'cellsexid=cellsexid.cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        'cellsexid': ['*.py'],
    },
    zip_safe=False,
    keywords="single-cell RNA-seq sex prediction machine-learning bioinformatics",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/CellSexID/issues",
        "Source": "https://github.com/yourusername/CellSexID",
        "Documentation": "https://github.com/yourusername/CellSexID/blob/main/README.md",
    },
)
