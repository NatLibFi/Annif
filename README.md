<img src="https://annif.org/static/img/annif-RGB.svg" width="150">

[![DOI](https://zenodo.org/badge/100936800.svg)](https://zenodo.org/badge/latestdoi/100936800)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI/CD](https://github.com/NatLibFi/Annif/actions/workflows/cicd.yml/badge.svg)](https://github.com/NatLibFi/Annif/actions/workflows/cicd.yml)
[![codecov](https://codecov.io/gh/NatLibFi/Annif/branch/main/graph/badge.svg)](https://codecov.io/gh/NatLibFi/Annif)
[![Code Climate](https://codeclimate.com/github/NatLibFi/Annif/badges/gpa.svg)](https://codeclimate.com/github/NatLibFi/Annif)
[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/NatLibFi/Annif/badges/quality-score.png?b=main)](https://scrutinizer-ci.com/g/NatLibFi/Annif/?branch=main)
[![codebeat badge](https://codebeat.co/badges/7a8ef539-0094-48b8-84c2-c413b4a50d57)](https://codebeat.co/projects/github-com-natlibfi-annif-main)
[![CodeQL](https://github.com/NatLibFi/Annif/actions/workflows/codeql.yml/badge.svg)](https://github.com/NatLibFi/Annif/actions/workflows/codeql.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=NatLibFi_Annif&metric=alert_status)](https://sonarcloud.io/dashboard?id=NatLibFi_Annif)
[![docs](https://readthedocs.org/projects/annif/badge/?version=latest)](https://annif.readthedocs.io/en/latest/index.html)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Annif is an automated subject indexing toolkit. It was originally created as
a statistical automated indexing tool that used metadata from the
[Finna.fi](https://finna.fi) discovery interface as a training corpus.

This repo contains a rewritten production version of Annif based on the
[prototype](https://github.com/osma/annif). It is a work in progress, but
already functional for many common tasks.

[Finto AI](https://ai.finto.fi/) is a service based on Annif; see the [source code for Finto AI](https://github.com/NatLibFi/FintoAI).

# Basic install

Annif is developed and tested on Linux. If you want to run Annif on Windows or Mac OS, the recommended way is to use Docker (see below) or a Linux virtual machine.

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

## Shell compeletions
Annif supports tab-key completion in bash, zsh and fish shells for commands and options
and project id, vocabulary id and path parameters.

To enable the completion support in your current terminal session use `annif completion`
command with the option according to your shell to produce the completion script and
source it. For example, run

    source <(annif completion --bash)

To enable the completion support in all new sessions first add the completion script in
your home directory:

    annif completion --bash > ~/.annif-complete.bash

Then make the script to be automatically sourced for new terminal sessions by adding the
following to your `~/.bashrc` file (or in some [alternative startup
file](https://www.gnu.org/software/bash/manual/html_node/Bash-Startup-Files.html)):

    source ~/.annif-complete.bash

For details and usage for other shells see
[Click documentation](https://click.palletsprojects.com/en/8.1.x/shell-completion/).
# Docker install

You can use Annif as a pre-built Docker container. Please see the
[wiki documentation](https://github.com/NatLibFi/Annif/wiki/Usage-with-Docker)
for details.

# Development install

A development version of Annif can be installed by cloning the [GitHub
repository](https://github.com/NatLibFi/Annif).
[Poetry](https://python-poetry.org/) is used for managing dependencies and virtual environment for the development version.

See [CONTRIBUTING.md](CONTRIBUTING.md) for information on [unit tests](CONTRIBUTING.md#unit-tests), [code style](CONTRIBUTING.md#code-style), [development flow](CONTRIBUTING.md#development-flow) etc. details that are useful when participating in Annif development.

## Installation and setup

Clone the repository.

Switch into the repository directory.

Install [pipx](https://pypa.github.io/pipx/) and Poetry if you don't have them. First pipx:

    python3 -m pip install --user pipx
    python3 -m pipx ensurepath

Open a new shell, and then install Poetry:

    pipx install poetry

Poetry can be installed also without pipx: check the [Poetry documentation](https://python-poetry.org/docs/master/#installation).

Create a virtual environment and install dependencies:

    poetry install

By default development dependencies are included. Use option `-E` to install dependencies for selected optional features (`-E "extra1 extra2"` for multiple extras), or install all of them with `--all-extras`. By default the virtual environment directory is not under the project directory, but there is a [setting for selecting this](https://python-poetry.org/docs/configuration/#virtualenvsin-project).

Enter the virtual environment:

    poetry shell

You will also need NLTK data files:

    python -m nltk.downloader punkt

Start up the application:

    annif

# Getting help

Many resources are available:

 * [Usage documentation in the wiki](https://github.com/NatLibFi/Annif/wiki)
 * [Annif tutorial](https://github.com/NatLibFi/Annif-tutorial) for learning to use Annif
 * [annif-users](https://groups.google.com/forum/#!forum/annif-users) discussion forum
 * [Internal API documentation](https://annif.readthedocs.io) on ReadTheDocs
 * [annif.org](https://annif.org) project web site

# Publications / How to cite

Two articles about Annif have been published in peer-reviewed Open Access
journals. The software itself is also archived on Zenodo and
has a [citable DOI](https://doi.org/10.5281/zenodo.5654173).

## Citing the software itself

See "Cite this repository" in the details of the repository.

## Annif articles
<ul>
<li>
Suominen, O.; Inkinen, J.; Lehtinen, M., 2022.
Annif and Finto AI: Developing and Implementing Automated Subject Indexing.
JLIS.It, 13(1), pp. 265–282. URL:
https://www.jlis.it/index.php/jlis/article/view/437
<details>
<summary>See BibTex</summary>

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
</details>
</li>
<li>
Suominen, O.; Koskenniemi, I, 2022.
Annif Analyzer Shootout: Comparing text lemmatization methods for automated subject indexing.
Code4Lib Journal, (54). URL:
https://journal.code4lib.org/articles/16719
<details>
<summary>See BibTex</summary>

    @article{suominen2022analyzer,
      title={Annif Analyzer Shootout: Comparing text lemmatization methods for automated subject indexing},
      author={Suominen, Osma and Koskenniemi, Ilkka},
      journal={Code4Lib J.},
      number={54},
      year={2022},
      url={https://journal.code4lib.org/articles/16719},
    }
</details>
</li>
<li>
Suominen, O., 2019. Annif: DIY automated subject indexing using multiple
algorithms. LIBER Quarterly, 29(1), pp.1–25. DOI:
https://doi.org/10.18352/lq.10285
<details>
<summary>See BibTex</summary>

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
</details>
</li>
</ul>

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
