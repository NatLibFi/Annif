import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='annif',
    version='0.37.2',
    url='https://github.com/NatLibFi/Annif',
    author='Osma Suominen',
    author_email='osma.suominen@helsinki.fi',
    description='Automated subject indexing and classification tool',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'connexion[swagger-ui]',
        'swagger_ui_bundle',
        'flask-cors',
        'click-log',
        'nltk',
        'gensim',
        'sklearn',
        'fasttextmirror',
        'rdflib'],
    entry_points={
        'console_scripts': ['annif=annif.cli:cli']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ]
)
