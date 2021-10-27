import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tssearch",
    version="0.0.1",
    author="Fraunhofer Portugal",
    description="Library for time series subsequence search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['scipy', 'pandas', 'matplotlib', 'numpy'],
)
