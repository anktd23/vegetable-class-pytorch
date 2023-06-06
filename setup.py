from setuptools import setup,find_packages
from typing import List

HYPEN_E_DOT = '-e .'
requirements = []
def get_requirements(file_path:str)->List[str]:
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","")for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='vegetable-classification',
    author='Aniket Dumbre',
    author_email='anktd23@gmail.com',
    version='0.0.1',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)