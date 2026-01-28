#!/bin/bash

# Pre-pull Ollama models before starting FastAPI
echo "Initializing Ollama models..."

# Get Ollama host from env or use default
OLLAMA_HOST=${OLLAMA_HOST:-"http://ollama:11434"}

# Use Python to handle HTTP requests (more reliable than curl in containers)
python3 << 'EOF'
import time
import urllib.request
import json
import sys

ollama_host = "http://ollama:11434"
max_attempts = 60  # 2 minutes total
attempt = 0

# Wait for Ollama to be ready
print("Waiting for Ollama service...")
while attempt < max_attempts:
    try:
        with urllib.request.urlopen(f"{ollama_host}/api/tags", timeout=2) as response:
            if response.status == 200:
                print("✓ Ollama is ready")
                break
    except Exception as e:
        attempt += 1
        if attempt % 10 == 0:
            print(f"Waiting for Ollama... (attempt {attempt}/{max_attempts})")
        time.sleep(2)
else:
    print("✗ Warning: Ollama did not respond after timeout, proceeding anyway...")

# Models to pre-pull
models = ["gemma3:270m", "nomic-embed-text"]

print("\nPre-pulling required models...")
for model in models:
    print(f"Checking if model {model} exists...")
    try:
        # Check if model exists
        with urllib.request.urlopen(f"{ollama_host}/api/tags", timeout=5) as response:
            tags = json.loads(response.read().decode())
            model_names = [m.get("name", "") for m in tags.get("models", [])]
            
            if any(model in name for name in model_names):
                print(f"✓ Model {model} already exists")
                continue
    except Exception as e:
        print(f"Could not check model status: {e}")
    
    # Pull the model
    print(f"Pulling model {model}...")
    try:
        pull_data = json.dumps({"name": model}).encode()
        req = urllib.request.Request(
            f"{ollama_host}/api/pull",
            data=pull_data,
            method="POST",
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=600) as response:
            # Stream response lines
            while True:
                line = response.readline()
                if not line:
                    break
                try:
                    data = json.loads(line.decode())
                    if "status" in data:
                        status = data.get("status", "")
                        if "downloading" in status.lower() or "pulling" in status.lower():
                            digest = data.get("digest", "")[-12:] if data.get("digest") else ""
                            print(f"  {status} {digest}")
                except:
                    pass
            print(f"✓ Model {model} pulled successfully")
    except Exception as e:
        print(f"Warning: Failed to pull {model}: {e}")

print("\nOllama initialization complete!")
EOF

echo "Starting FastAPI..."
exec python -c 'import sys; sys.path.insert(0, "."); from src.main import app; import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8000)'
