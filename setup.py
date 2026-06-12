from setuptools import setup, find_packages

setup(
    name="qstkl",
    version="0.1.0",
    description="Quantum Semantic Toolkit and Library: Complex-valued neural networks and quantum semantic methods (pure numpy core, optional npcpy extensions).",
    author="Christopher Agostino",
    author_email="info@npcworldwi.de",
    url="https://npcworldwi.de",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "npcpy>=1.3.33",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="quantum semantics complex neural networks toolkit",
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    project_urls={
        "Company": "https://enpisi.com",
    },
)
