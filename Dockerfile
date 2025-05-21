FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y

ENV MPLCONFIGDIR=/tmp
ENV HOME=/app
ENV XDG_CACHE_HOME=/app/.cache
ENV MPLCONFIGDIR=/tmp
ENV HF_HOME=/app/hf_cache

RUN mkdir -p /app/hf_cache /app/.cache && chmod -R 777 /app/hf_cache /app/.cache

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/iragca/ml-final-project-streamlit-app.git

WORKDIR /app/ml-final-project-streamlit-app

RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    python3-dev
RUN pip install uv
RUN uv sync

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["uv", "run", "streamlit", "run", "main.py", "--server.port=7860", "--server.address=0.0.0.0"]
