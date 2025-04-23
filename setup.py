# setup.py
from setuptools import setup, find_packages

setup(
    name='workload_vae',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas=1.0.0',
        'numpy=1.18.0',
        'torch=1.6.0',
        'scikit-learn=0.22.0',
        'matplotlib=3.1.0',
        'seaborn=0.10.0'
    ],
    author='Your Name',
    author_email='your@email.com',
    description='A Python package for processing workload CSVs using a Variational Autoencoder (VAE)',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='textmarkdown',
    url='httpsgithub.comyourusernameworkload_vae',
    classifiers=[
        'Programming Language  Python  3',
        'License  OSI Approved  MIT License',
    ],
)
