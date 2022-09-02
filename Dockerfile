FROM python:3.8-slim-bullseye
LABEL maintainer="Juho Inkinen <juho.inkinen@helsinki.fi>"
SHELL ["/bin/bash", "-c"]

ARG optional_dependencies="fasttext voikko pycld3 fasttext nn omikuji yake spacy"
ARG POETRY_VIRTUALENVS_CREATE=false

# Install system dependencies needed at runtime:
RUN apt-get update && \
	if [[ $optional_dependencies =~ "voikko" ]]; then \
		apt-get install -y --no-install-recommends \
			libvoikko1 \
			voikko-fi; \
	fi && \
	# curl for Docker healthcheck and rsync for model transfers:
	apt-get install -y --no-install-recommends curl rsync && \
	rm -rf /var/lib/apt/lists/* /usr/include/*

WORKDIR /Annif
RUN pip install --upgrade pip poetry --no-cache-dir && \
	pip install poetry

COPY pyproject.toml setup.cfg README.md LICENSE.txt CITATION.cff projects.cfg.dist /Annif/

# First round of installation for Docker layer caching:
RUN echo "Installing dependencies for optional features: $optional_dependencies" \
	&& poetry install -E "$optional_dependencies"

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

# Download nltk data
RUN python -m nltk.downloader punkt -d /usr/share/nltk_data

WORKDIR /annif-projects

# Switch user to non-root:
RUN groupadd -g 998 annif_user && \
    useradd -r -u 998 -g annif_user annif_user && \
    chmod -R a+rX /Annif && \
    chown -R annif_user:annif_user /annif-projects
USER annif_user

CMD annif
