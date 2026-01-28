FROM python:3.12-slim-bookworm

# Install dependencies
# RUN apt update && apt upgrade -y && rm -rf /var/lib/apt/lists/*
# RUN apt update && apt install -y --no-install-recommends curl ca-certificates && \
#     rm -rf /var/lib/apt/lists/*

# Copy the application files
# Copy the application files
COPY src/ /fastAPI/src/
COPY config/ /fastAPI/config/

# Copy requirements and entrypoint script
COPY requirements.txt /temp_files/requirements.txt

# Create a non-root user and group
RUN useradd -m appuser
# Set permissions so appuser owns what it needs
RUN chown -R appuser:appuser /fastAPI /tmp

# Get the build arguments, with a default values
ARG ENV_TYPE=deploy
# Set the environment variable in the container
ENV ENV_TYPE=${ENV_TYPE}

# Note: We use the main server/llm_system/core/llm.py which is already copied above
# It uses ChatOllama with hardcoded "http://ollama:11434" which works for both dev and prod


# Switch to non-root user
USER appuser
WORKDIR /fastAPI
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Install dependencies:
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --prefer-binary -r /temp_files/requirements.txt

# Switch back to root briefly to clean up
USER root
RUN rm -rf /temp_files && pip cache purge
USER appuser

# EXPOSE PORTS
# dev -> EXPOSE 8000, 11434

# Start entrypoint
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]