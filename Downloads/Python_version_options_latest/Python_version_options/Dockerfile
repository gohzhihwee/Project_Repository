# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — builder
# Installs all Python dependencies into an isolated --user prefix so that only
# the compiled wheels (not pip/setuptools metadata) need to be copied into the
# final runtime image.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Prevents Python from writing .pyc files and forces stdout/stderr to be
# unbuffered — both matter for a long-running batch job.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Copy only the dependency manifest first so Docker's layer cache lets us skip
# the expensive pip install step on every code-only change.
COPY requirements.txt .

# Upgrade pip, then install into the user-local prefix (~/.local).
# --no-cache-dir keeps the layer small.
# --no-warn-script-location silences the harmless "scripts not on PATH" note.
RUN pip install --upgrade pip --quiet --no-cache-dir \
 && pip install \
        --quiet \
        --no-cache-dir \
        --user \
        --no-warn-script-location \
        -r requirements.txt


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime
# A minimal image that carries only the installed packages and the application
# source.  No build tools, no pip, no root execution.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Tell Python where the --user packages from the builder stage live.
    PYTHONPATH=/home/appuser/.local/lib/python3.12/site-packages \
    PATH=/home/appuser/.local/bin:$PATH \
    # Use the non-interactive Agg backend — no display is available in Docker.
    MPLBACKEND=Agg

# Create a non-root user and the output directory in one layer.
RUN useradd --create-home --uid 1000 --shell /bin/bash appuser \
 && mkdir -p /app/output \
 && chown -R appuser:appuser /app

# Pull the compiled packages from the builder stage into the non-root user's
# local package directory.
COPY --from=builder /root/.local /home/appuser/.local

WORKDIR /app

# Copy all application source files.
# Deliberately listed individually (not COPY . .) so that adding unrelated
# files to the project directory does not invalidate this layer.
COPY AlgorithmImports.py \
     calibration.py \
     convmethods.py \
     data_loader.py \
     ensemble.py \
     main.py \
     metrics.py \
     models.py \
     modelstrats.py \
     visualization.py \
     ./

# Hand ownership of the copied files to the non-root user.
RUN chown -R appuser:appuser /app

USER appuser


# Default entry point.  Docker Compose may override CMD via `command:`.
ENTRYPOINT ["python", "-u", "main.py"]
