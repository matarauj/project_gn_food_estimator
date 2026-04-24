# =============================================================================
# Dockerfile — GN Food Estimator on HuggingFace Spaces
# =============================================================================
# All app files live in /app/src/ to match the HF Spaces Docker convention.
#
# Key environment variables:
#   HF_HOME=/tmp/huggingface  — redirects model cache to a writable path
#                               (HF Spaces restricts writes to home dir)
#   STREAMLIT_HOME=/tmp/.streamlit — same reason for Streamlit's own files
# =============================================================================

FROM python:3.12-slim

WORKDIR /app

# System dependencies needed by OpenCV headless
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code into src/
COPY src/ ./src/

# Redirect all cache and config writes to /tmp (writable in HF Spaces)
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface
ENV STREAMLIT_HOME=/tmp/.streamlit
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose Streamlit's default port
EXPOSE 8501

# Health check so HF knows when the app is ready
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the app — main file is src/app.py
ENTRYPOINT ["streamlit", "run", "src/app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]