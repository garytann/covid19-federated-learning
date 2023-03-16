from setuptools import find_packages, setup

setup(
    name="covidlus",
    version="0.0.1",
    python_requires=">=3.7.0",
    install_requires=[
        "argparse",
        "numpy",
        "pandas",
        "pre-commit",
        "scikit-learn",
        "scipy",
        "seaborn",
        "setuptools==59.5.0",
        "tensorboard",
        "torch",
        "torchvision",
        "tqdm",
        "umap-learn",
        "opacus",
    ],
    author="SIT-UOG",
    author_email="unknown",
    packages=find_packages(),
    include_package_data=True,
)