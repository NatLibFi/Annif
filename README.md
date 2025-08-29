<img src="https://annif.org/static/img/annif-RGB.svg" width="150">

[![DOI](https://zenodo.org/badge/100936800.svg)](https://zenodo.org/badge/latestdoi/100936800)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Container image](https://img.shields.io/badge/container_image-quay.io-blue.svg)](https://quay.io/repository/natlibfi/annif)
[![CI/CD](https://github.com/NatLibFi/Annif/actions/workflows/cicd.yml/badge.svg)](https://github.com/NatLibFi/Annif/actions/workflows/cicd.yml)
[![codecov](https://codecov.io/gh/NatLibFi/Annif/branch/main/graph/badge.svg)](https://codecov.io/gh/NatLibFi/Annif)
[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/NatLibFi/Annif/badges/quality-score.png?b=main)](https://scrutinizer-ci.com/g/NatLibFi/Annif/?branch=main)
[![Code Climate](https://codeclimate.com/github/NatLibFi/Annif/badges/gpa.svg)](https://codeclimate.com/github/NatLibFi/Annif)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/NatLibFi/Annif/badge)](https://securityscorecards.dev/viewer/?uri=github.com/NatLibFi/Annif)
[![CodeQL](https://github.com/NatLibFi/Annif/actions/workflows/codeql.yml/badge.svg)](https://github.com/NatLibFi/Annif/actions/workflows/codeql.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=NatLibFi_Annif&metric=alert_status)](https://sonarcloud.io/dashboard?id=NatLibFi_Annif)
[![docs](https://readthedocs.org/projects/annif/badge/?version=latest)](https://annif.readthedocs.io/en/latest/index.html)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Open in GitHub Codespaces](https://img.shields.io/static/v1?&label=Tutorial+in+Codespaces&message=Open&color=brightgreen&logo=github)](https://codespaces.new/NatLibFi/Annif-tutorial/tree/codespaces)

Annif is an automated subject indexing toolkit. It was originally created as
a statistical automated indexing tool that used metadata from the
[Finna.fi](https://finna.fi) discovery interface as a training corpus.

Annif provides [CLI commands](https://annif.readthedocs.io/en/stable/source/commands.html) for administration, and a [REST API](https://api.annif.org/v1/ui/) and web UI for end-users.

[Finto AI](https://ai.finto.fi/) is a service based on Annif;
see a [🤗 Hugging Face Hub collection](https://huggingface.co/collections/NatLibFi/annif-models-65b35fb98b7c508c8e8a1570) of the models that Finto AI uses.

This repository contains a rewritten production version of Annif based on the
[prototype](https://github.com/osma/annif).

# Basic install

Annif is developed and tested on Linux. If you want to run Annif on Windows or Mac OS, the recommended way is to use Docker (see below) or a Linux virtual machine.

You will need Python 3.10-3.13 to install Annif.

The recommended way is to install Annif from
[PyPI](https://pypi.org/project/annif/) into a virtual environment.

    python3 -m venv annif-venv
    source annif-venv/bin/activate
    pip install annif

Start up the application:

    annif

See [Getting Started](https://github.com/NatLibFi/Annif/wiki/Getting-started)
for basic usage instructions and
[Optional features and dependencies](https://github.com/NatLibFi/Annif/wiki/Optional-features-and-dependencies)
for installation instructions for e.g. fastText and Omikuji backends and for Voikko and spaCy analyzers.

## Shell compeletions
Annif supports tab-key completion in bash, zsh and fish shells for commands and options
and project id, vocabulary id and path parameters.
The completion functionality is not enabled after Annif installation;
get instructions for how to enable it by running

    annif completion --help

or see [this wiki page](https://github.com/NatLibFi/Annif/wiki/Command-line-interface#shell-compeletions).
# Docker install

You can use Annif as a pre-built Docker container image from [quay.io/natlibfi/annif](https://quay.io/repository/natlibfi/annif) repository. Please see the
[wiki documentation](https://github.com/NatLibFi/Annif/wiki/Usage-with-Docker)
for details.

# Demo install in Codespaces
Annif can be tried out in the [GitHub Codespaces](https://docs.github.com/en/codespaces). Just open a page for configuring a new codespace via the badge below, start the codespace from the green "Create codespace" button, and a terminal session will start in your browser. The environment will have Annif installed and the contents of the [Annif-tutorial](https://github.com/NatLibFi/Annif-tutorial) repository available.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/NatLibFi/Annif-tutorial/tree/codespaces)

# Development install

A development version of Annif can be installed by cloning the [GitHub
repository](https://github.com/NatLibFi/Annif).
[Poetry](https://python-poetry.org/) is used for managing dependencies and virtual environment for the development version; Poetry 2.0+ is required.

See [CONTRIBUTING.md](CONTRIBUTING.md) for information on [unit tests](CONTRIBUTING.md#unit-tests), [code style](CONTRIBUTING.md#code-style), [development flow](CONTRIBUTING.md#development-flow) etc. details that are useful when participating in Annif development.

## Installation and setup

Clone the repository.

Switch into the repository directory.

Install [pipx](https://pypa.github.io/pipx/) and Poetry if you don't have them. First pipx:

    python3 -m pip install --user pipx
    python3 -m pipx ensurepath

Open a new shell, and then install Poetry:

    pipx install poetry==2.*

Poetry can be installed also without pipx: check the [Poetry documentation](https://python-poetry.org/docs/master/#installation).

Create a virtual environment and install dependencies:

    poetry install

By default development dependencies are included. Use option `-E` to install dependencies for selected optional features (`-E "extra1 extra2"` for multiple extras), or install all of them with `--all-extras`. By default the virtual environment directory is not under the project directory, but there is a [setting for selecting this](https://python-poetry.org/docs/configuration/#virtualenvsin-project).

Enter the virtual environment:

    eval $(poetry env activate)

Start up the application:

    annif

# Getting help

Many resources are available:

 * [Usage documentation in the wiki](https://github.com/NatLibFi/Annif/wiki)
 * [Annif tutorial](https://github.com/NatLibFi/Annif-tutorial) for learning to use Annif
 * [annif-users](https://groups.google.com/forum/#!forum/annif-users) discussion forum; please use this as a channel for questions instead of personal e-mails to developers
 * [Internal API documentation](https://annif.readthedocs.io) on ReadTheDocs
 * [annif.org](https://annif.org) project web site

# Publications / How to cite

See below for some articles about Annif in peer-reviewed Open Access
journals. The software itself is also archived on Zenodo and
has a [citable DOI](https://doi.org/10.5281/zenodo.5654173).

## Citing the software itself

See "Cite this repository" in the details of the repository.

## Annif articles
<ul>
<li>
Suominen, O; Inkinen, J.; Lehtinen, M. 2025.
Annif at the GermEval-2025 LLMs4Subjects Task: Traditional XMTC Augmented by Efficient LLMs, pre-print.
https://arxiv.org/abs/2508.15877
<details>
<summary>See BibTex</summary>

    @misc{suominen2025annifgermeval2025,
      title={https://arxiv.org/abs/2508.15877},
      author={Osma Suominen and Juho Inkinen and Mona Lehtinen},
      year={2025},
      eprint={2508.15877},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.15877},
    }
</details>
</li>
<li>
Suominen, O; Inkinen, J.; Lehtinen, M. 2025.
Annif at SemEval-2025 Task 5: Traditional XMTC augmented by LLMs.
In Proceedings of the 19th International Workshop on Semantic Evaluation (SemEval-2025), pp. 2424–2431, Vienna, Austria. Association for Computational Linguistics.
https://aclanthology.org/2025.semeval-1.315/
https://arxiv.org/abs/2504.19675
<details>
<summary>See BibTex</summary>
    
    @misc{suominen2025annifsemeval2025task5,
      title={Annif at SemEval-2025 Task 5: Traditional XMTC augmented by LLMs},
      title = "Annif at {S}em{E}val-2025 Task 5: Traditional {XMTC} augmented by {LLM}s",
      author = "Suominen, Osma and Inkinen, Juho and Lehtinen, Mona",
      editor = "Rosenthal, Sara  and  Ros{\'a}, Aiala  and  Ghosh, Debanjan  and  Zampieri, Marcos",
      booktitle = "Proceedings of the 19th International Workshop on Semantic Evaluation (SemEval-2025)",
      month = jul,
      year = "2025",
      address = "Vienna, Austria",
      publisher = "Association for Computational Linguistics",
      url = "https://aclanthology.org/2025.semeval-1.315/",
      pages = "2424--2431",
      ISBN = "979-8-89176-273-2",
      # ArXiv 
      # year={2025},
      # eprint={2504.19675},
      # archivePrefix={arXiv},
      # primaryClass={cs.CL},
      # url={https://arxiv.org/abs/2504.19675},
    }
</details>
</li>
<li>
Inkinen, J.; Lehtinen, M.; Suominen, O., 2025.
Annif Users Survey: Understanding Usage and Challenges.
URL:
https://urn.fi/URN:ISBN:978-952-84-1301-1
<details>
<summary>See BibTex</summary>

    @misc{inkinen2025,
      title={Annif Users Survey: Understanding Usage and Challenges},
      author={Inkinen, Juho and Lehtinen, Mona and Suominen, Osma},
      series={The National Library of Finland. Reports and Studies},
      issn={2242–8119},
      isbn={978-952-84-1301-1},
      year={2025},
      url={URN:ISBN:978-952-84-1301-1},
    }
</details>
</li>
<li>
Golub, K.; Suominen, O.; Mohammed, A.; Aagaard, H.; Osterman, O., 2024.
Automated Dewey Decimal Classification of Swedish library metadata using Annif software.
Journal of Documentation, 80(5), pp. 1057-1079. URL:
https://doi.org/10.1108/JD-01-2022-0026
<details>
<summary>See BibTex</summary>

    @article{golub2024annif,
      title={Automated Dewey Decimal Classification of Swedish library metadata using Annif software},
      author={Golub, Koraljka and Suominen, Osma and Mohammed, Ahmed Taiye and Aagaard, Harriet and Osterman, Olof},
      journal={J. Doc.},
      year={2024},
      doi = {10.1108/JD-01-2022-0026},
      url={https://www.emerald.com/insight/content/doi/10.1108/JD-01-2022-0026},
    }
</details>
</li>
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

The code in this repository is licensed under Apache License 2.0, except for
the dependencies included under `annif/static/css` and `annif/static/js`,
which have their own licenses; see the file headers for details.

Please note that the [YAKE](https://github.com/LIAAD/yake) library is
licensed under [GPLv3](https://www.gnu.org/licenses/gpl-3.0.txt), while
Annif itself is licensed under the Apache License 2.0. It is commonly
accepted that the GPLv3 and Apache 2.0 licenses are compatible at least in
one direction (GPLv3 is more restrictive than the Apache License); obviously
it also depends on the legal environment. The Annif developers make no legal
claims - we simply provide the software and allow the user to install
optional extensions if they consider it appropriate. Depending on legal
interpretation, the terms of the GPL (for example the requirement to publish
corresponding source code when publishing an executable application) may be
considered to apply to the whole of Annif+extensions if you decide to
install the optional YAKE dependency.
