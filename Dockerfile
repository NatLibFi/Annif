FROM python:3.6-slim

# Using old pip version because --no-cache-dir doesn't seem to work in 19.1.1 
RUN pip install --upgrade pip==18.1 \
	&& pip install pipenv --no-cache-dir

COPY . /Annif
# TODO Copy only needed files for pipenv install in this layer
#COPY Pipfile Pipfile.lock setup.py setup.cfg /Annif/

WORKDIR /Annif

# TODO Handle occasional timeout in nltk.downloader leading failed build
# TODO Disable caching in pipenv, maybe EXPORT PIP_NO_CACHE_DIR=false 
RUN pipenv install --system --deploy --ignore-pipfile --dev \
	&& python -m nltk.downloader punkt


## Install optional dependencies
# Voikko
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		libvoikko1 \
		voikko-fi \
    && pip install --no-cache-dir \
    	annif[voikko] \
	&& rm -rf /var/lib/apt/lists/*

# fasttext
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		build-essential \
	&& pip install --no-cache-dir \
		cython \
		fasttextmirror \
	&& rm -rf /var/lib/apt/lists/*

# Vowpal Wabbit. Using old VW because 8.5 links to wrong Python version
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		libboost-program-options-dev\
		zlib1g-dev \
		libboost-python-dev \
	&& pip install --no-cache-dir \
		vowpalwabbit==8.4 \
	&& rm -rf /var/lib/apt/lists/*

CMD annif
