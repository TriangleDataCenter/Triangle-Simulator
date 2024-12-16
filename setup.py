import os

from setuptools import find_packages, setup


# Function to read the requirements from requirements.txt
def read_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


# Function to read the long description from README.md
def read_long_description():
    with open("README.md", encoding="utf-8") as f:
        return f.read()


setup(
    name="Triangle",  # Replace with your desired package name
    version="0.1.0",  # Start with 0.1.0 and update as needed
    description="A brief description of your project.",  # Provide a short description
    long_description=read_long_description(),
    long_description_content_type="text/markdown",  # Specifies that README is in Markdown
    author="Minghui Du",
    author_email="your.email@example.com",  # Replace with your email
    url="https://github.com/yourusername/TriangleProject",  # Replace with your project's URL
    license="MIT",  # Replace with your project's license
    packages=find_packages(),  # Automatically find packages in the directory
    include_package_data=True,  # Include package data as specified in MANIFEST.in
    package_data={
        "": [
            "Figures/*.png",
            "GWData/*.npy",
            "OrbitData/**/*.dat",
            "OrbitData/**/*.hdf5",
        ],
    },
    install_requires=read_requirements(),  # Dependencies from requirements.txt
    classifiers=[
        "Programming Language :: Python :: 3",  # Specify the Python versions you support
        "License :: OSI Approved :: MIT License",  # Replace if different
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Specify the Python versions you support
)
