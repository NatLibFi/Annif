import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='annif',
    version='0.53.2',
    url='https://github.com/NatLibFi/Annif',
    author='Osma Suominen',
    author_email='osma.suominen@helsinki.fi',
    description='Automated subject indexing and classification tool',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=[
        'connexion[swagger-ui]',
        'swagger_ui_bundle',
        'flask',
        'flask-cors',
        'click==8.0.*',
        'click-log',
        'joblib==1.0.1',
        'nltk',
        'gensim==4.0.*',
        'scikit-learn==0.24.2',
        'scipy==1.5.4',
        'rdflib>=4.2,<6.0',
        'gunicorn',
        'numpy==1.19.*',
        'optuna==2.8.0',
        'stwfsapy==0.2.*'
    ],
    tests_require=['py', 'pytest', 'requests'],
    extras_require={
        'fasttext': ['fasttext==0.9.2'],
        'voikko': ['voikko'],
        'vw': ['vowpalwabbit==8.10.2'],
        'nn': ['tensorflow-cpu==2.5.0', 'lmdb==1.2.1'],
        'omikuji': ['omikuji==0.3.*'],
        'yake': ['yake==0.4.5'],
        'dev': [
            'codecov',
            'pytest-cov',
            'pytest-watch',
            'pytest-flask',
            'pytest-flake8',
            'swagger-tester',
            'bumpversion',
            'responses',
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
