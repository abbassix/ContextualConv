from setuptools import setup, find_packages

setup(
    name="contextual-conv",
    version="0.1.0",
    author="Mehdi Abbassi",
    author_email="mehdi.n.abbassi@gmail.com",
    description="PyTorch convolutional layers with global context conditioning",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/abbassix/ContextualConv",
    license="GPLv3",
    packages=find_packages(),
    install_requires=[
        "pytest",
        # Don't include torch to allow manual install (CPU or CUDA)
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
