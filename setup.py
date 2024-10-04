import os
from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file.
with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="knowledge-graph-similarity",
    author="Blue Brain Project, EPFL",
    use_scm_version={
        "relative_to": __file__,
        "write_to": "similarity_tools/version.py",
        "write_to_template": "__version__ = '{version}'\n",
    },
    description="Tools for performing registration of data for similarity-based inference.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="ontology knowledge graph data science",
    packages=find_packages(),
    python_requires=">=3.7",
    include_package_data=True,
    setup_requires=[
        "setuptools_scm",
    ],
    install_requires=[
        f"knowledge-graph-inference",
        "bluegraph@git+https://github.com/BlueBrain/BlueGraph",
        "neurom",
        "tmd"
    ],
    extras_require={
        "dev": [
            "pytest==7.2.1",
            "pytest-cov==4.1.0",
            "pytest-bdd",
            "pytest-mock==3.3.1",
            "codecov",
            "flake8"
        ]
    },
    classifiers=[
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3 :: Only",
        "Natural Language :: English",
    ]
)
