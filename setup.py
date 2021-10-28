from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='img_recognition',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    url="https://github.com/saratbhargava/img_recognition",
    version='0.0.1',
    description='Implement DL based Image recognition systems from scatch',
    author='Sarat Chinni',
    author_email='sarat.chinni@outlook.com',
    license='MIT',
)
