from setuptools import setup

setup(
    name='binary-evolution',
    version='0.1.dev',
    packages=['binary-evolution'],
    url='https://github.com/mwbub/binary-evolution',
    author='Mathew Bub',
    author_email='mathew.bub@gmail.com',
    description='Package for evolving the orbits of compact binaries modelled '
                'as Keplerian rings perturbed by an external potential',
    install_requires=['numpy', 'scipy', 'astropy', 'galpy']
)
