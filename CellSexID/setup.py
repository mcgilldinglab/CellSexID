from setuptools import setup, find_packages

setup(
    name="cellsexid",
    version="1.0.0",
    description="Predict biological sex from single-cell RNA-seq data.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/CellSexID",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scanpy>=1.9.0",
        "xgboost>=1.6.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "anndata>=0.8.0",
        "seaborn>=0.11.0",
    ],
    entry_points={
        "console_scripts": [
            "cellsexid-run=cellsexid.cli:main",
        ]
    },
    python_requires=">=3.7",
)
