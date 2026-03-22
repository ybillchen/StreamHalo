from setuptools import setup, find_packages

setup(
    name="streamhalo",
    version="0.1.0",
    description="Mock stellar halo generator with tidal streams",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "astropy>=5.0",
    ],
)
