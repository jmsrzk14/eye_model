FROM python:3.13-slim

# Install system dependencies untuk OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libxcb1 \
    libfontconfig1 \
    libice6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements terlebih dahulu (untuk caching layer)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh aplikasi
COPY . .

# Set default PORT
ENV PORT=8000

# Expose port
EXPOSE 8000

# Command untuk run aplikasi - gunakan exec form dengan sh
ENTRYPOINT ["sh", "-c"]
CMD ["gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --max-requests 100"]