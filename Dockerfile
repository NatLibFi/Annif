FROM python:3.6-slim


## Update Using old pip version because --no-cache-dir doesn't seem to work in 19.1.1
RUN pip install --upgrade pip==18.1 \
	&& rm -rf /root/.cache/pip*/*


## Install optional dependencies
# Voikko
RUN apt-get update && apt-get install -y --no-install-recommends \
		libvoikko1 \
		voikko-fi \
    && pip install --no-cache-dir \
    	annif[voikko] \
	&& rm -rf /var/lib/apt/lists/* /usr/include/*

# fasttext
RUN apt-get update && apt-get install -y --no-install-recommends \
		build-essential \
	&& pip install --no-cache-dir \
		cython \
    	fasttextmirror \
	&& apt-get remove --auto-remove -y build-essential \
	&& rm -rf /var/lib/apt/lists/* /usr/include/*

# Vowpal Wabbit. Using old VW because 8.5 links to wrong Python version
RUN apt-get update && apt-get install -y --no-install-recommends \
		build-essential \
		libboost-program-options-dev \
		zlib1g-dev \
		libboost-python-dev \
    && pip install --no-cache-dir \
		vowpalwabbit==8.4 \
	&& apt-get remove --auto-remove -y \
		build-essential \
		zlib1g-dev \
	&& rm -rf /var/lib/apt/lists/* /usr/include/* \
	&& rm -rf /root/.cache/pip*/* \
	&& rm -rf /usr/lib/python2.7*


## Install Annif
# Files needed by pipenv install:
COPY Pipfile Pipfile.lock README.md setup.py /Annif/
WORKDIR /Annif

# TODO Handle occasional timeout in nltk.downloader leading failed build
RUN pip install pipenv --no-cache-dir \
	&& pipenv install --system --deploy --ignore-pipfile \
	&& python -m nltk.downloader punkt \
	&& pip uninstall -y pipenv \
	&& rm -rf /root/.cache/pip*/*

COPY annif annif

CMD annif
