FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY src/ src/

RUN cd ../
RUN mkdir -p data/datasets/daily
RUN mkdir -p data/datasets/hourly
RUN mkdir -p data/datasets/variations/
RUN mkdir -p data/synapse/
RUN mkdir -p plots/
RUN mkdir -p results/

CMD ["tail", "-f", "/dev/null"]
