from setuptools import find_packages,setup
from typing import List
HYPEN_E_DOT='-e .'
def get_requirementes(file_path:'str')->List[str]:
    '''this function return list of requirementes'''
    requirementes=[]
    with open(file_path) as file_obj:
        requirementes=file_obj.readlines()
        requirementes=[req.replace("\n","") for req in requirementes]
        if HYPEN_E_DOT in requirementes:
            requirementes.remove(HYPEN_E_DOT)
    return requirementes

setup(
    name='mlproject',
    version='0.0.1',
    author='nikhilpawar2000',
    author_email='nikhilpawar08012000@gmail.com',
    packages=find_packages(),
    install_requires=get_requirementes('requirementes.txt')
)