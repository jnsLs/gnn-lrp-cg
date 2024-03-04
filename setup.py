from setuptools import setup, find_packages

setup(
    name='gnn_lrp_qc',
    version='0.2.0',
    packages=find_packages("src"),
    package_dir={"": "src"},
)