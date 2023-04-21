FROM python:3

WORKDIR /usr/src/app

RUN pip install --no-cache-dir pymongo

COPY scripts/generate_mongo_config_file.py .

ENTRYPOINT ["python", "generate_mongo_config_file.py"]
