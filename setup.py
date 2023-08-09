from setuptools import setup, find_packages

setup(
    name='poli_baselines',
    version='0.0.1',
    install_requires=[
        'numpy',
        'click',
    ],
    packages=find_packages(
        # where='src',  # '.' by default
        exclude=['tests'],  # empty by default
    )
)

