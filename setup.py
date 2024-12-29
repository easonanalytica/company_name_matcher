from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the contents of your requirements file
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="company_name_matcher",
    version="0.1.1",
    author="Eason Suen",
    author_email=None,
    description="A library for matching and comparing company names using a fine-tuned sentence transformer model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url= "https://github.com/easonanalytica/company_name_matcher",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
)
