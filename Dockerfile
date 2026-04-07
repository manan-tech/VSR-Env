# VSR-Env Dockerfile
# Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6

FROM python:3.11-slim

# Install curl for healthcheck (Requirement 14.5)
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies (Requirement 14.2)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY vsr_env/ ./vsr_env/
COPY pyproject.toml .
COPY openenv.yaml .
COPY inference.py .
COPY README.md .

# Install the package
RUN pip install --no-cache-dir -e .

# Expose port 7860 (Hugging Face Requirement)
EXPOSE 7860

# Healthcheck endpoint (Requirement 14.5)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Enable the built-in web interface for Hugging Face Spaces / judges
ENV ENABLE_WEB_INTERFACE=true

# Run uvicorn server (Requirement 14.4)
CMD ["uvicorn", "vsr_env.server.app:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]