# Declare global ARG
ARG USERNAME=manatal

FROM python:3.10-slim AS base

# Update the base layer
RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get clean

# Setup non-root user
ARG USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME
RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Set the default user to walnut
USER $USERNAME

# Setup PATH for installed binary
ENV PATH "/home/$USERNAME/.local/bin:/home/$USERNAME/app/.venv/bin/:$PATH"

# Create directory
RUN mkdir -p /home/$USERNAME/app
WORKDIR /home/$USERNAME/app

FROM base AS builder

# Reinitiate global ARG and set the default user to walnut
ARG USERNAME
USER $USERNAME

# Install poetry
RUN pip install poetry

# Copy the required files to install requirements
COPY poetry.lock pyproject.toml poetry.toml ./

# Install the requirements into .venv
RUN poetry config installer.max-workers 10
RUN poetry install --no-root --only main,gpu

FROM base AS app

# Reinitiate global ARG and set the default user to walnut
ARG USERNAME
USER $USERNAME

# Copy .venv from builder
COPY --from=builder /home/$USERNAME/app/.venv ./.venv

# Set the Production Environment
ENV MANATAL_ENV production

# Copy all the required source files
COPY ai/ ./ai
COPY api/ ./api
COPY data/ ./data
COPY models/ ./models
COPY helper.py .
COPY main.py .
COPY config.py .
COPY schema.py .
COPY .env.$MANATAL_ENV .

# Declare PORT
ENV PORT 8080
EXPOSE $PORT

CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
