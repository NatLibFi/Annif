FROM python:3.8-slim-bullseye AS builder

LABEL maintainer="Juho Inkinen <juho.inkinen@helsinki.fi>"

# Install fastText, which needs to be built, and therefore also some system packages:
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		build-essential \
	&& pip install --no-cache-dir \
		fasttext==0.9.2


FROM python:3.8-slim-bullseye

COPY --from=builder /usr/local/lib/python3.8 /usr/local/lib/python3.8

# Install system dependencies needed at runtime:
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		libvoikko1 \
		voikko-fi \
	# For Docker healthcheck:
	&& apt-get install -y --no-install-recommends curl \
	&& rm -rf /var/lib/apt/lists/* /usr/include/*

WORKDIR /Annif
RUN pip install --upgrade pip --no-cache-dir

# Install all optional dependencies:
COPY setup.py README.md LICENSE.txt projects.cfg.dist /Annif/
RUN pip install .[dev,voikko,pycld3,fasttext,nn,omikuji,vw,yake] --no-cache-dir

# Download nltk data (handle occasional timeout in with 3 tries):
RUN for i in 1 2 3; do python -m nltk.downloader punkt -d /usr/share/nltk_data && break || sleep 1; done

# Install Annif by copying source and make the installation editable:
COPY annif /Annif/annif
COPY tests /Annif/tests
RUN pip install -e .

WORKDIR /annif-projects

# Switch user to non-root:
RUN groupadd -g 998 annif_user \
    && useradd -r -u 998 -g annif_user annif_user \
    && chown -R annif_user:annif_user /annif-projects
USER annif_user

CMD annif
