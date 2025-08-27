from setuptools import setup, find_packages

setup(
    name="openlrm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your package dependencies here, for example:
        # 'numpy',
        # 'requests',
    ],
    author="Sam Bahrami",
    author_email="bahrami.sam@gmail.com",
    description="A brief description of the openlrm package",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/SamBahrami/PluckeRF",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',  # Specify the minimum Python version required
)
