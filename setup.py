import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='annif',
    version='0.43.0',
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
        'flask',
        'flask-cors',
        'click',
        'click-log',
        'joblib==0.13.*',
        'nltk',
        'gensim==3.8.*',
        'scikit-learn==0.21.*',
        'rdflib',
        'numpy==1.17.*',
    ],
    tests_require=['py', 'pytest', 'requests'],
    extras_require={
        'fasttext': ['fasttext', 'fasttextmirror==0.8.22'],
        'voikko': ['voikko'],
        'vw': ['vowpalwabbit==8.7.*'],
        'nn': ['tensorflow==2.0.*'],
    },
    entry_points={
        'console_scripts': ['annif=annif.cli:cli']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ]
)
