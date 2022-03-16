import setuptools
import pip
from subprocess import call

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gsa",
    version="0.0.1",
    author="Wonpyo Park",
    author_email="wppark.pio@gmail.com",
    description="Graph Self Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lenscloth/GraphSelfAttention",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.7",
    install_requires=["ogb", "rdkit-pypi"],
)
