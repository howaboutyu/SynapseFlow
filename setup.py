from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="synapseflow",
    version="1.0.0",
    author="Haobo Yu",
    author_email="haoboyu806@gmail.com",
    description="A computational neural science package for SynapseFlow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/howaboutyu/SynapseFlow",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        # Add any dependencies required by SynapseFlow here
        "numpy>=1.0.0",
        "pandas>=1.0.0",
    ],
    python_requires=">=3.6",
)
