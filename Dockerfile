FROM python:3.10-slim-bookworm
LABEL org.opencontainers.image.authors="grp-natlibfi-annif@helsinki.fi"
SHELL ["/bin/bash", "-c"]

ARG optional_dependencies="voikko fasttext nn omikuji yake spacy stwfsa"
ARG POETRY_VIRTUALENVS_CREATE=false

# Install system dependencies needed at runtime:
RUN apt-get update && apt-get upgrade -y && \
	if [[ $optional_dependencies =~ "voikko" ]]; then \
		apt-get install -y --no-install-recommends \
			libvoikko1 \
			voikko-fi; \
	fi && \
	# Install rsync for model transfers:
	apt-get install -y --no-install-recommends rsync && \
	rm -rf /var/lib/apt/lists/* /usr/include/*

WORKDIR /Annif
RUN pip install --upgrade pip poetry --no-cache-dir && \
	pip install poetry

COPY pyproject.toml setup.cfg README.md LICENSE.txt CITATION.cff projects.cfg.dist /Annif/

# First round of installation for Docker layer caching:
RUN echo "Installing dependencies for optional features: $optional_dependencies" \
	&& poetry install -E "$optional_dependencies" \
	&& rm -rf /root/.cache/pypoetry  # No need for cache because of poetry.lock

# Download nltk data
RUN python -m nltk.downloader punkt -d /usr/share/nltk_data

# Download spaCy models, if the optional feature was selected
ARG spacy_models=en_core_web_sm
RUN if [[ $optional_dependencies =~ "spacy" ]]; then \
		for model in $(echo $spacy_models | tr "," "\n"); do \
			python -m spacy download $model; \
		done; \
	fi

# Second round of installation with the actual code:
COPY annif /Annif/annif
COPY tests /Annif/tests
RUN poetry install -E "$optional_dependencies"

WORKDIR /annif-projects
RUN annif completion --bash >> /etc/bash.bashrc  # Enable tab completion

# Switch user to non-root:
RUN groupadd -g 998 annif_user && \
    useradd -r -u 998 -g annif_user annif_user && \
    chmod -R a+rX /Annif && \
    mkdir -p /Annif/tests/data /.cache/huggingface /Annif/projects.d && \
    chown -R annif_user:annif_user /annif-projects /Annif/tests/data
USER annif_user

CMD annif
