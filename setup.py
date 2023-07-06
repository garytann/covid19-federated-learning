# from setuptools import find_packages, setup

# setup(
#     name="covidlus",
#     version="0.0.1",
#     python_requires=">=3.7.0",
#     install_requires=[
#         "argparse",
#         "numpy",
#         "pandas",
#         "pre-commit",
#         "scikit-learn",
#         "scipy",
#         "seaborn",
#         "setuptools==59.5.0",
#         "tensorboard",
#         "torch",
#         "torchvision",
#         "tqdm",
#         "umap-learn",
#         "opacus",
#         "fastapi",
#         "uvicorn"
#     ],
#     author="SIT-UOG",
#     author_email="unknown",
#     packages=find_packages(),
#     include_package_data=True,
# )

"""Install package."""
from setuptools import setup, find_packages
setup(
    name='pocovidnet',
    version='0.0.1',
    description=(
        'Keras implementation of COVID19 detection models from POCUS data'
    ),
    long_description=open('README.md').read(),
    # url='https://github.com/jannisborn/medimg_covid_detecter',
    author='Gary Tan SIT-UOG',
    author_email='2002870@sit.singaporetech.edu.sg',
    install_requires=[
        'numpy', 'tensorflow', 'scikit-learn', 'matplotlib', 'imutils',
        'opencv-contrib-python', 'flask'
    ],
    packages=find_packages('.'),
    zip_safe=False,
)