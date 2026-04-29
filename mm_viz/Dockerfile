FROM python:3.11-slim

WORKDIR /app

# Install python deps once at build time (fast container startup).
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only the viz app (code is also bind-mounted in compose for dev iteration).
COPY mm_viz /app/mm_viz

EXPOSE 8501

CMD ["streamlit", "run", "mm_viz/app.py", "--server.address=0.0.0.0", "--server.port=8501"]

