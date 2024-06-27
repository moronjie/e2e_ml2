# ml app as a package 
from setuptools import find_packages,setup

def get_requirements(path:str)->list[str]:
    '''
    this function will return the list of requirements
    '''
    requirements_list=[]
    if path:
        with open(path) as file_obj:
            requirements_list=file_obj.readlines()
            requirements_list=[req.replace("\n","") for req in requirements_list]
            if "-e ." in requirements_list:
                requirements_list.remove("-e .")
    return requirements_list

setup(
    name="myMLProject",
    version="0.0.1",
    author="Moro Njie",
    author_email="njiemoro2@gmail.com",
    packages=find_packages(),
    install_requires= get_requirements("requirements.txt")
)