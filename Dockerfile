FROM python:3.7-slim AS builder

LABEL maintainer="Juho Inkinen <juho.inkinen@helsinki.fi>"

## Install optional dependencies:
RUN apt-get update \
	## Voikko:
	&& apt-get install -y --no-install-recommends \
		libvoikko1 \
		voikko-fi \
	&& pip install --no-cache-dir \
		annif[voikko] \
	## fasttext:
	&& apt-get install -y --no-install-recommends \
		build-essential \
	&& pip install --no-cache-dir \
		cython \
		fasttextmirror \
	## Vowpal Wabbit
	&& apt-get install -y --no-install-recommends \
		libboost-program-options-dev \
		zlib1g-dev \
		libboost-python-dev \
		cmake \
		libboost-system-dev \
		libboost-thread-dev \
		libboost-test-dev \
	&& ln -sf /usr/lib/x86_64-linux-gnu/libboost_python-py35.a \
		/usr/lib/x86_64-linux-gnu/libboost_python3.a \
	&& ln -sf /usr/lib/x86_64-linux-gnu/libboost_python-py35.so \
		/usr/lib/x86_64-linux-gnu/libboost_python3.so \
	&& pip install --no-cache-dir \
		vowpalwabbit



FROM python:3.7-slim

COPY --from=builder /usr/local/lib/python3.7 /usr/local/lib/python3.7

## Dependencies needed at runtime:
RUN apt-get update \
	## Voikko:
	&& apt-get install -y --no-install-recommends \
		libvoikko1 \
		voikko-fi \
	&& pip install --no-cache-dir \
		annif[voikko] \
	## Vowpal Wabbit
	&& apt-get install -y --no-install-recommends \
		libboost-program-options1.62.0 \
		libboost-python1.62.0 \
		libboost-system1.62.0 \
	&& pip install --no-cache-dir \
		vowpalwabbit \
	## Clean up:
	&& rm -rf /var/lib/apt/lists/* /usr/include/* \
	&& rm -rf /root/.cache/pip*/*


## Install Annif:
# Files needed by pipenv install:
COPY Pipfile README.md setup.py /Annif/
WORKDIR /Annif

# Handle occasional timeout in nltk.downloader with 3 tries
RUN pip install pipenv --no-cache-dir \
	&& pipenv install --system --skip-lock \
	&& for i in 1 2 3; do python -m nltk.downloader punkt -d /usr/share/nltk_data && break || sleep 1; done \
	&& pip uninstall -y pipenv \
	&& rm -rf /root/.cache/pip*/*


COPY annif annif
COPY projects.cfg.dist projects.cfg.dist

WORKDIR /annif-projects


# Switch user to non-root:
RUN groupadd -g 998 annif_user \
    && useradd -r -u 998 -g annif_user annif_user \
    && chown -R annif_user:annif_user /annif-projects
USER annif_user


CMD annif
