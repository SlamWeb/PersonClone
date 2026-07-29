FROM node:20-bookworm-slim@sha256:2cf067cfed83d5ea958367df9f966191a942351a2df77d6f0193e162b5febfc0 AS web-builder

WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build


FROM python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models/huggingface \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    PERSONAFORGE_WEB_DIST=/app/web/dist

WORKDIR /app

# The Web runtime also executes server-side author jobs. Install CPU PyTorch
# explicitly to avoid pulling a much larger CUDA runtime, then add Playwright
# Chromium for the authenticated crawler fallback.
#
# A temporary package skeleton lets Docker cache third-party dependencies by
# pyproject.toml. Editing application source no longer reinstalls PyTorch.
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p src/personaforge \
    && touch src/personaforge/__init__.py \
    && printf '# PersonaForge\n' > README.md \
    && python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install ".[crawler,index,web]" \
    && python -m playwright install --with-deps chromium \
    && rm -rf src README.md build

COPY README.md ./
COPY src/ ./src/
RUN python -m pip uninstall --yes personaforge \
    && python -m pip install --no-cache-dir --no-build-isolation --no-deps . \
    && pf --version

COPY --from=web-builder /build/web/dist ./web/dist

RUN groupadd --gid 10001 personaforge \
    && useradd --uid 10001 --gid personaforge --create-home personaforge \
    && mkdir -p /app/data /app/models \
    && chown -R personaforge:personaforge /app

USER personaforge

VOLUME ["/app/data", "/app/models"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["pf", "web", "--host", "0.0.0.0", "--port", "8000", "--data-dir", "/app/data", "--embedding-device", "cpu", "--no-fp16"]
