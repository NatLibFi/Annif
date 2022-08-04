import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='annif',
    version='0.58.0',
    url='https://annif.org',
    project_urls={
        'Source': 'https://github.com/NatLibFi/Annif',
        'Documentation': 'https://github.com/NatLibFi/Annif/wiki',
    },
    author='Osma Suominen',
    author_email='osma.suominen@helsinki.fi',
    description='Automated subject indexing and classification tool',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=[
        'connexion[swagger-ui]==2.14.*',
        'swagger_ui_bundle',
        'flask>=1.0.4,<3',
        'flask-cors',
        'click==8.0.*',
        'click-log',
        'joblib==1.1.0',
        'nltk',
        'gensim==4.2.*',
        'scikit-learn==1.1.1',
        'scipy==1.8.*',
        'rdflib>=4.2,<7.0',
        'gunicorn',
        'numpy==1.23.*',
        'optuna==2.10.*',
        'stwfsapy==0.3.*',
        'python-dateutil',
        'tomli==2.0.*',
        'simplemma==0.7.*'
    ],
    tests_require=['py', 'pytest', 'requests'],
    extras_require={
        'fasttext': ['fasttext==0.9.2'],
        'voikko': ['voikko'],
        'nn': ['tensorflow-cpu==2.9.1', 'lmdb==1.3.0'],
        'omikuji': ['omikuji==0.5.*'],
        'yake': ['yake==0.4.5'],
        'pycld3': ['pycld3'],
        'spacy': ['spacy==3.3.*'],
        'dev': [
            'codecov',
            'coverage<=6.2',
            'pytest-cov',
            'pytest-watch',
            'pytest-flask',
            'pytest-flake8',
            'flake8<5',
            'bumpversion',
            'autopep8'
        ]
    },
    entry_points={
        'console_scripts': ['annif=annif.cli:cli']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ]
)
