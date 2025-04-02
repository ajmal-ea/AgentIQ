from setuptools import setup, find_packages

setup(
    name="audience_research",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "aiq.plugins": [
            "audience_research=audience_research.register",
        ],
    },
) 