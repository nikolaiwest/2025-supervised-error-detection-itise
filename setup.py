from setuptools import find_packages, setup

setup(
    name="supervised-error-detection",
    version="1.0.0",
    description="Supervised error detection in dataset s04 for variations in assembly conditions",
    author="Nikolai West",
    author_email="nikolai.west@tu-dortmund.de",
    url="https://github.com/nikolaiwest/2025-supervised-error-detection-itise",
    packages=find_packages(),
    install_requires=[
        "pyscrew",
        "sktime",
        "numba",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "mlflow",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
