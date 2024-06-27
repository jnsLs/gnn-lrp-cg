from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = list(
        filter(lambda x: "#" not in x, (line.strip() for line in f))
    )

setup(
    name="gnn_lrp_qc",
    version="0.2.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=install_requires,
)
