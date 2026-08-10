# Use a Python 3.12 + uv image (Debian bookworm-slim)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
LABEL org.opencontainers.image.authors="grp-natlibfi-annif@helsinki.fi"
SHELL ["/bin/bash", "-c"]

ARG optional_dependencies="voikko fasttext nn omikuji yake spacy stwfsa torch-cpu"

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

# Copy only project metadata first to maximize Docker layer caching
COPY pyproject.toml setup.cfg README.md LICENSE.txt CITATION.cff projects.cfg.dist /Annif/

# First round: install dependencies only (no project), with selected extras.
RUN extras=(); \
    for e in ${optional_dependencies}; do extras+=(--extra "$e"); done; \
    uv sync --no-install-project "${extras[@]}"

# Download nltk data
RUN uv run --no-sync python -m nltk.downloader punkt_tab -d /usr/share/nltk_data

# Second round: add source and install the actual project (editable by default)
COPY annif /Annif/annif
COPY tests /Annif/tests
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
RUN annif completion --bash >> /etc/bash.bashrc  # Enable tab completion

# Set up working dir & non-root user for running annif commands
WORKDIR /annif-projects
RUN groupadd -g 998 annif_user && \
    useradd -r -u 998 -g annif_user annif_user && \
    chmod -R a+rX /Annif/* && \
    mkdir -p /Annif/tests/data /Annif/projects.d && \
    chown -R annif_user:annif_user /annif-projects /Annif/tests/data
USER annif_user
ENV HF_HOME="/tmp"

ENV GUNICORN_CMD_ARGS="--worker-class uvicorn.workers.UvicornWorker"

CMD annif
