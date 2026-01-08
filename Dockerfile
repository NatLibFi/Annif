# Use a Python 3.12 + uv image (Debian bookworm-slim)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
LABEL org.opencontainers.image.authors="grp-natlibfi-annif@helsinki.fi"
SHELL ["/bin/bash", "-c"]

ARG optional_dependencies="voikko fasttext nn omikuji yake spacy stwfsa"

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

RUN groupadd -g 998 annif_user && \
    useradd -m -u 998 -g annif_user annif_user && \
    chown -R annif_user:annif_user /Annif
USER annif_user

# Copy only project metadata first to maximize Docker layer caching
COPY --chown=annif_user:annif_user pyproject.toml setup.cfg README.md LICENSE.txt CITATION.cff projects.cfg.dist /Annif/

# First round: install dependencies only (no project), with selected extras.
RUN extras=(); \
    for e in ${optional_dependencies}; do extras+=(--extra "$e"); done; \
    uv sync --no-install-project "${extras[@]}"

# Download nltk data
RUN uv run --no-sync python -m nltk.downloader punkt_tab

# Second round: add source and install the actual project (editable by default)
COPY --chown=annif_user:annif_user annif /Annif/annif
COPY --chown=annif_user:annif_user tests /Annif/tests
RUN extras=(); \
    for e in ${optional_dependencies}; do extras+=(--extra "$e"); done; \
    uv sync "${extras[@]}"

# Download spaCy models only if 'spacy' extra is selected
ARG spacy_models=en_core_web_sm
RUN if [[ $optional_dependencies =~ "spacy" ]]; then \
        for model in $(echo "$spacy_models" | tr "," "\n"); do \
            uv run --no-sync python -m spacy download "$model"; \
        done; \
    fi

# Make virtualenv executables available to shell and entrypoint
ENV PATH="/Annif/.venv/bin:${PATH}"

# Enable Annif bash completion (now available on PATH)
WORKDIR /annif-projects
RUN annif completion --bash >> ~/.bashrc

ENV GUNICORN_CMD_ARGS="--worker-class uvicorn.workers.UvicornWorker"

CMD annif
