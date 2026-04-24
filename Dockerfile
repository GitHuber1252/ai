FROM python:3.11-slim

WORKDIR /app

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=
ARG PIP_TRUSTED_HOST=

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}
ENV PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --retries 20 --timeout 180 -r requirements.txt \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1 \
    || (echo "Primary index failed, retrying with default PyPI URL" \
    && PIP_INDEX_URL=https://pypi.org/simple python -m pip install --no-cache-dir --retries 20 --timeout 180 -r requirements.txt \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1)

COPY app.py .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
