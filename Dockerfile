# 1. Base image با PyTorch CPU آماده
FROM pytorch/pytorch:2.2.0-cpu

# 2. Set working directory
WORKDIR /app

# 3. سیستم نیازمندی‌ها برای hazm
RUN apt-get update && apt-get install -y build-essential python3-dev && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements
COPY requirements.txt .

# 5. Upgrade pip و نصب dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy app
COPY . .

# 7. Run main
CMD ["python", "main.py"]
