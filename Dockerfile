# 1. Base image با PyTorch CPU آماده
FROM pytorch/pytorch:2.2.0-cpu

# 2. Set working directory
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

# 7. Run main
CMD ["python", "main.py"]
