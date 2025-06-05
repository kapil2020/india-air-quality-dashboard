# 1. Use a minimal Python 3.12 base image
FROM python:3.12-slim

# 2. Install system dependencies required for geopandas, shapely, cartopy, tabula-py, etc.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev \
    libgeos-dev libproj-dev \
    default-jre \
    build-essential \
    curl ca-certificates unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 3. Set environment variables so pip and geopandas can find GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV PYTHONUNBUFFERED=1

# 4. Set working directory
WORKDIR /app

# 5. Copy only requirements first (for Docker layer caching)
COPY requirements.txt .

# 6. Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of the app code
COPY . .

# 8. Expose Streamlit port
EXPOSE 8501

# 9. Start the Streamlit app, using Vercel's injected $PORT
CMD streamlit run app.py --server.port "$PORT" --server.address 0.0.0.0
