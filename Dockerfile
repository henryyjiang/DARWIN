# 1. Start from a base image with Python installed
FROM python:3.11-slim

# 2. Set a working directory inside the container
WORKDIR /app

RUN pip install --no-cache-dir torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu126


# 3. Copy requirements.txt first (for caching installs)
COPY nanogpt/requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project files
COPY . .

# 6. Run your app by default
CMD ["python", "main/main_controller.py"]
