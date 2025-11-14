# Use a Python base image for ARM architecture
FROM python:3.9-slim-bullseye

# Set the working directory
WORKDIR /app


# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the default command to run the real-time classifier
CMD ["python", "src/real_time_classifier.py"]
