FROM python:3.6

#FROM python:3.6-slim
#FROM python:3-alpine
#FROM vaultvulp/pipenv-alpine

# TODO Using old pip version because --no-cache-dir doesn't seem to work in 19.1.1 
RUN pip install pipenv --upgrade pip==18.1

COPY . /Annif
WORKDIR /Annif

#RUN pipenv install --system --deploy --ignore-pipfile
RUN pipenv install --system --deploy --ignore-pipfile --dev \
	&& python -m nltk.downloader punkt

# Voikko optional dependency
RUN apt-get update && apt-get install -y libvoikko1 voikko-fi \
    && pip install annif[voikko] --no-cache-dir

# fasttext optional dependency
# TODO Why cython is needed?
RUN pip install cython fasttextmirror --no-cache-dir

# Vowpal Wabbit fastext optional dependency
RUN apt-get install -y libboost-program-options-dev libboost-python-dev zlib1g-dev
RUN pip install vowpalwabbit --no-cache-dir

CMD annif
#ENTRYPOINT ["annif"]

# TODO EXPORT PIP_NO_CACHE_DIR=false for disabling caching in pipenv