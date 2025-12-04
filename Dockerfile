# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
# By copying the requirements file first, we can take advantage of Docker's layer caching.
# The following RUN command will only be re-executed if the requirements.txt file changes.
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# This layer is cached and will not be re-run unless requirements.txt changes.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir hazm --no-dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
# This layer will only be re-built if the application's code changes.
COPY . .

# Run main.py when the container launches
CMD ["python", "main.py"]
