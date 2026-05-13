from setuptools import setup, find_packages

setup(
    name="falkor-irac",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "falkordb>=0.5.0",
        "pymupdf>=1.24.0",
        "pdfminer.six>=20221105",
        "requests>=2.31.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "rich>=13.7.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
    ],
)
