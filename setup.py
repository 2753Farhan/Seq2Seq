from setuptools import setup, find_packages

setup(
    name="seq2seq-codegen",
    version="1.0.0",
    author="BSSE Student",
    description="Seq2Seq Models for Text-to-Python Code Generation",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "nltk>=3.8.0",
        "datasets>=2.12.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
)