FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a non-root user to run the application
RUN useradd -m platon
USER platon

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the dashboard port
EXPOSE 8050

# Command to run the application
CMD ["python", "run_platon_light.py"]
