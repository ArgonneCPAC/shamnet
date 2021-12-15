from setuptools import setup, find_packages


PACKAGENAME = "shamnet"
VERSION = "0.1.0"


setup(
    name=PACKAGENAME,
    version=VERSION,
    author="Andrew Hearin",
    author_email="ahearin@anl.gov",
    description="Differentiable abundance matching",
    long_description="Differentiable abundance matching",
    install_requires=["numpy", "jax", "corrfunc", "h5py", "numba"],
    packages=find_packages(),
    package_data={"shamnet": ["data/*.h5", "tests/testing_data/*.h5"]},
    url="https://github.com/ArgonneCPAC/shamnet",
)
