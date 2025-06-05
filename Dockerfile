# Use slim Python base
FROM python:3.12-slim

# System dependencies for geopandas, shapely, cartopy, etc.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev \
    libgeos-dev libproj-dev \
    default-jre \
    build-essential \
    curl unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# GDAL environment for geopandas
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy dependencies first for cache
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Streamlit port
EXPOSE 8501

# Start the Streamlit app using Vercel's $PORT
CMD streamlit run app.py --server.port $PORT --server.address 0.0.0.0
