FROM ultralytics/ultralytics:latest
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir roboflow