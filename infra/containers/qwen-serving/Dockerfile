FROM public.ecr.aws/pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/opt/ml/model/.cache

WORKDIR /opt/program

COPY requirements.txt /opt/program/requirements.txt
RUN pip install --no-cache-dir -r /opt/program/requirements.txt

COPY inference.py /opt/program/inference.py

ENV SAGEMAKER_PROGRAM=inference.py
