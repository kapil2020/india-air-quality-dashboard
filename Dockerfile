# 1) Base image: slim Python 3.12
FROM python:3.12-slim

# 2) Install system‐level dependencies needed by GeoPandas, Cartopy, OSMnx, Tabula‐Py, etc.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      gdal-bin libgdal-dev \
      libgeos-dev libproj-dev \
      default-jre \
      build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 3) Make sure GDAL headers can be found during Python package builds
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# 4) Set working directory inside the container
WORKDIR /app

# 5) Copy only requirements.txt for layer caching
COPY requirements.txt /app/

# 6) Install all Python dependencies (from requirements.txt)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7) Copy the rest of your application code
COPY . /app/

# 8) Expose Streamlit’s default port in the container
EXPOSE 8501

# 9) Use shell‐form CMD so that $PORT is expanded at runtime by Vercel
CMD streamlit run app.py --server.port "$PORT" --server.address 0.0.0.0
