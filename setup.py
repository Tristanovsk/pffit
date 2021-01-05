
from setuptools import setup, find_packages
exec(open('pffit/version.py').read())

setup(
    name=__package__,
    version=__version__,
    packages=find_packages(exclude=['build']),
    package_data={'':['*.so','*h','*angles*','*.txt','*.csv']},
    #     # If any package contains *.txt files, include them:
    #     '': ['*.txt'],
    #     'lut': ['data/lut/*.nc'],
    #     'aux': ['data/aux/*']
    # },
    include_package_data=True,

    url='',
    license='MIT',
    author='T. Harmel',
    author_email='tristan.harmel@gmail.com',
    description='tools to simulate and fit hydrosol''s phase functions',

    # Dependent packages (distributions)
    install_requires=['pandas','numpy','xarray','lmfit',
                      'matplotlib'],

    entry_points={
          'console_scripts': [
              #'pffit_exe : pffit.main'
          ]}
)
