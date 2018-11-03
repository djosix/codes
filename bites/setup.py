import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bites",
    version="0.0.9",
    author="djosix",
    author_email="toregenerate@gmail.com",
    description="Operating bytes made easy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/djosix/bites",

    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
)
