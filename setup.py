from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='brainmnist',
      version="0.1",
      description="Brain Mnist Package",
      license="MIT",
      author="<Names>",
      author_email="<mails>",
      url="https://github.com/biancasama/Brain-MNIST",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      include_package_data=True,
      zip_safe=False)
