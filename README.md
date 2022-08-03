<img src="https://annif.org/static/img/annif-RGB.svg" width="150">

[![DOI](https://zenodo.org/badge/100936800.svg)](https://zenodo.org/badge/latestdoi/100936800)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI/CD](https://github.com/NatLibFi/Annif/actions/workflows/cicd.yml/badge.svg)](https://github.com/NatLibFi/Annif/actions/workflows/cicd.yml)
[![codecov](https://codecov.io/gh/NatLibFi/Annif/branch/master/graph/badge.svg)](https://codecov.io/gh/NatLibFi/Annif)
[![Code Climate](https://codeclimate.com/github/NatLibFi/Annif/badges/gpa.svg)](https://codeclimate.com/github/NatLibFi/Annif)
[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/NatLibFi/Annif/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/NatLibFi/Annif/?branch=master)
[![codebeat badge](https://codebeat.co/badges/e496f151-93db-4f0e-9e30-bc3339e58ca4)](https://codebeat.co/projects/github-com-natlibfi-annif-master)
[![BCH compliance](https://bettercodehub.com/edge/badge/NatLibFi/Annif?branch=master)](https://bettercodehub.com/)
[![LGTM: Python](https://img.shields.io/lgtm/grade/python/g/NatLibFi/Annif.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/NatLibFi/Annif/context:python)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=NatLibFi_Annif&metric=alert_status)](https://sonarcloud.io/dashboard?id=NatLibFi_Annif)
[![docs](https://readthedocs.org/projects/annif/badge/?version=latest)](https://annif.readthedocs.io/en/latest/index.html)

Annif is an automated subject indexing toolkit. It was originally created as
a statistical automated indexing tool that used metadata from the
[Finna.fi](https://finna.fi) discovery interface as a training corpus.

This repo contains a rewritten production version of Annif based on the
[prototype](https://github.com/osma/annif). It is a work in progress, but
already functional for many common tasks.

# Basic install

You will need Python 3.8+ to install Annif.

The recommended way is to install Annif from
[PyPI](https://pypi.org/project/annif/) into a virtual environment.

    python3 -m venv annif-venv
    source annif-venv/bin/activate
    pip install annif

You will also need NLTK data files:

    python -m nltk.downloader punkt

Start up the application:

    annif

See [Getting Started](https://github.com/NatLibFi/Annif/wiki/Getting-started)
in the wiki for more details.

# Docker install

You can use Annif as a pre-built Docker container. Please see the 
[wiki documentation](https://github.com/NatLibFi/Annif/wiki/Usage-with-Docker)
for details.

# Development install

A development version of Annif can be installed by cloning the [GitHub
repository](https://github.com/NatLibFi/Annif).

## Installation and setup

Clone the repository.

Switch into the repository directory.

Create and activate a virtual environment (optional, but highly recommended):

    python3 -m venv venv
    . venv/bin/activate

Install dependencies (including development) and make the installation editable:

    pip install .[dev]
    pip install -e .

You will also need NLTK data files:

    python -m nltk.downloader punkt

Start up the application:

    annif

## Unit tests

Run `. venv/bin/activate` to enter the virtual environment and then run `pytest`.
To have the test suite watch for changes in code and run automatically, use
pytest-watch by running `ptw`.

# Getting help

Many resources are available:

 * [Usage documentation in the wiki](https://github.com/NatLibFi/Annif/wiki)
 * [Annif tutorial](https://github.com/NatLibFi/Annif-tutorial) for learning to use Annif
 * [annif-users](https://groups.google.com/forum/#!forum/annif-users) discussion forum
 * [Internal API documentation](https://readthedocs.org/projects/annif/) on ReadTheDocs
 * [annif.org](http://annif.org) project web site

# Publications / How to cite

Two articles about Annif have been published in peer-reviewed Open Access
journals. The software itself is also archived on Zenodo and
has a [citable DOI](https://doi.org/10.5281/zenodo.5654173).

## Citing the software itself

See "Cite this repository" in the details of the repository.

## Annif articles

Suominen, O.; Inkinen, J.; Lehtinen, M., 2022. 
Annif and Finto AI: Developing and Implementing Automated Subject Indexing.
JLIS.It, 13(1), pp. 265–282. URL:
[https://www.jlis.it/index.php/jlis/article/view/437](https://www.jlis.it/index.php/jlis/article/view/437)

    @article{suominen2022annif,
      title={Annif and Finto AI: Developing and Implementing Automated Subject Indexing},
      author={Suominen, Osma and Inkinen, Juho and Lehtinen, Mona},
      journal={JLIS.it},
      volume={13},
      number={1},
      pages={265--282},
      year={2022},
      doi = {10.4403/jlis.it-12740},
      url={https://www.jlis.it/index.php/jlis/article/view/437},
    }

Suominen, O., 2019. Annif: DIY automated subject indexing using multiple
algorithms. LIBER Quarterly, 29(1), pp.1–25. DOI:
[https://doi.org/10.18352/lq.10285](https://doi.org/10.18352/lq.10285)

    @article{suominen2019annif,
      title={Annif: DIY automated subject indexing using multiple algorithms},
      author={Suominen, Osma},
      journal={{LIBER} Quarterly},
      volume={29},
      number={1},
      pages={1--25},
      year={2019},
      doi = {10.18352/lq.10285},
      url = {https://doi.org/10.18352/lq.10285}
    }

# License

The code in this repository is licensed under Apache License 2.0, except for the
dependencies included under `annif/static/css` and `annif/static/js`,
which have their own licenses, see the file headers for details.
Please note that the [YAKE](https://github.com/LIAAD/yake) library is licended
under [GPLv3](https://www.gnu.org/licenses/gpl-3.0.txt), while Annif is
licensed under the Apache License 2.0. The licenses are compatible, but
depending on legal interpretation, the terms of the GPLv3 (for example the
requirement to publish corresponding source code when publishing an executable
application) may be considered to apply to the whole of Annif+Yake if you
decide to install the optional Yake dependency.
