FROM tensorflow/tensorflow:latest-devel-gpu
FROM jupyter/minimal-notebook

RUN pip install mlflow
RUN pip install pandas
RUN pip install matplotlib

WORKDIR /app

COPY . .

