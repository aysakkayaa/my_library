from setuptools import setup, find_packages

setup(
    name="my_library",
    version="0.1.0",
    author="Ayşegül Akkaya",
    author_email="ayswasinthere@gmail.com",
    description="NLP ve Qdrant ile ilgili işlemleri kolaylaştıran bir Python kütüphanesi.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aysakkayaa/my_library",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "nltk",
        "pandas",
        "sentence-transformers",
        "qdrant-client",
    ],
)
