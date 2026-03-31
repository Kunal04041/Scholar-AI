# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Start script to run both services
RUN echo "#!/bin/bash\n\
uvicorn app.main:app --host 0.0.0.0 --port 8000 & \n\
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 \n\
" > /app/start.sh

RUN chmod +x /app/start.sh

# Run the application
CMD ["/app/start.sh"]
