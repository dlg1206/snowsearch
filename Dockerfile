FROM python:3.13-slim-trixie AS snowsearch
LABEL maintainer="Derek Garcia <dgarcia2@hawaii.edu>"
LABEL version=1.0.0
LABEL name="snowsearch"
# hardcode service endpoints - designed to be used inside stack
ENV SS_GROBID_SERVER=http://grobid:8070
ENV SS_OLLAMA_HOST=ollama
ENV BOLT_URI=bolt://abstract_db
ENV EMBEDDING_MODEL=all-MiniLM-L6-v2
WORKDIR /snowsearch
# setup dependencies
COPY requirements.txt .
RUN pip install --prefer-binary --no-cache -r requirements.txt
# create user
RUN useradd -m snowsearch
USER snowsearch
# cache embedding model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$EMBEDDING_MODEL')"
# copy code
COPY snowsearch snowsearch
# launch cli
ENTRYPOINT [ "python3", "snowsearch" ]
CMD [ "-h" ]