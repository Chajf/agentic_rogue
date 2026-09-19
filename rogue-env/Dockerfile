FROM debian:bookworm-slim AS rogue-builder

ARG ROGUE_REPOSITORY=https://github.com/lcn2/rogue5.4.git
ARG ROGUE_COMMIT=631bdd551cee75e0afc340a71dd5a207a3006204

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git libncurses-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none "${ROGUE_REPOSITORY}" /src/rogue \
    && git -C /src/rogue checkout --detach "${ROGUE_COMMIT}" \
    && test "$(git -C /src/rogue rev-parse HEAD)" = "${ROGUE_COMMIT}"
RUN make -C /src/rogue clobber all ROGUEDIR=/var/lib/rogue

FROM python:3.12-slim-bookworm AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libncurses6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 rogue \
    && install -d -o rogue -g rogue /var/lib/rogue

COPY --from=rogue-builder /src/rogue/rogue /usr/local/bin/rogue
WORKDIR /app
COPY pyproject.toml ./
COPY rogue_env ./rogue_env
RUN pip install --no-cache-dir .

USER rogue
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import json,urllib.request; data=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')); raise SystemExit(data['status'] != 'healthy')"
CMD ["uvicorn", "rogue_env.api:app", "--host", "0.0.0.0", "--port", "8000"]

