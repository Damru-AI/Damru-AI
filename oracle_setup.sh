#!/bin/bash
# =============================================================================
# DAMRU ORACLE CLOUD AUTO-SETUP
# =============================================================================
# Ek baar run karo -- sab kuch automatic ho jayega:
#   - llama.cpp server (Damru GGUF local inference -- NO API NEEDED)
#   - SearXNG (self-hosted Google -- NO API NEEDED)
#   - Curious Engine (Damru khud seekhta hai)
#   - Kaggle auto-trigger
#   - HuggingFace auto-push
#   - Systemd services (reboot pe bhi auto-start)
#
# HOW TO RUN:
#   1. Oracle Cloud Free Tier VM banao (Ubuntu 22.04 ARM, 4 OCPU, 24GB RAM)
#   2. SSH karo: ssh ubuntu@<YOUR_ORACLE_IP>
#   3. Ye file upload karo ya paste karo
#   4. chmod +x oracle_setup.sh && sudo bash oracle_setup.sh
#   5. Ek baar creds dalo -- phir kabhi mat kholna!
# =============================================================================
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

log() { echo -e "\n\e[32m[DAMRU-SETUP]\e[0m $*"; }
err() { echo -e "\n\e[31m[ERROR]\e[0m $*"; }

log "=================================================="
log " DAMRU ORACLE AUTO-SETUP  (API-FREE STACK)"
log "=================================================="

# ---- 0. Collect secrets once ------------------------------------------------
if [ ! -f /etc/damru/secrets.env ]; then
    mkdir -p /etc/damru && chmod 700 /etc/damru
    echo ""
    echo "=== Enter your credentials (one time only) ==="
    read -rp "HuggingFace Token (HF_TOKEN): " HF_TOKEN
    read -rp "Damru GGUF repo (e.g. Damaru-ai/damru-gguf): " GGUF_REPO
    read -rp "Kaggle Username: " KAGGLE_USER
    read -rp "Kaggle API Key: " KAGGLE_KEY
    read -rp "Your Email (for cron alerts): " ALERT_EMAIL

    cat > /etc/damru/secrets.env <<EOF
HF_TOKEN=${HF_TOKEN}
GGUF_REPO=${GGUF_REPO}
KAGGLE_USERNAME=${KAGGLE_USER}
KAGGLE_KEY=${KAGGLE_KEY}
ALERT_EMAIL=${ALERT_EMAIL}
DAMRU_MODEL_FILE=damru-model-q4_k_m.gguf
LLAMA_PORT=8080
SEARXNG_PORT=8888
CURIOUS_ENGINE_PORT=9000
EOF
    chmod 600 /etc/damru/secrets.env
    log "Secrets saved to /etc/damru/secrets.env"
fi

source /etc/damru/secrets.env

# ---- 1. System deps ---------------------------------------------------------
log "[1/8] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
    git cmake make g++ curl wget python3 python3-pip python3-venv \
    docker.io docker-compose screen tmux htop jq \
    build-essential libssl-dev libffi-dev \
    nodejs npm aria2 yt-dlp

systemctl enable docker && systemctl start docker
usermod -aG docker ubuntu 2>/dev/null || true

# ---- 2. llama.cpp (ARM-optimized, NO API needed) ----------------------------
log "[2/8] Building llama.cpp (ARM NEON optimized)..."
if [ ! -d /opt/llama.cpp ]; then
    git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp
    cd /opt/llama.cpp
    # ARM-optimized build
    cmake -B build -DLLAMA_NATIVE=ON -DLLAMA_ARM_FMA=ON -DLLAMA_F16C=ON
    cmake --build build --config Release -j$(nproc)
    log "llama.cpp built!"
fi

# ---- 3. Download Damru GGUF model -------------------------------------------
log "[3/8] Downloading Damru GGUF model from HF..."
mkdir -p /opt/damru/models
if [ ! -f "/opt/damru/models/${DAMRU_MODEL_FILE}" ]; then
    # huggingface-cli download
    pip3 install -q huggingface_hub
    python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='${GGUF_REPO}',
    filename='${DAMRU_MODEL_FILE}',
    token='${HF_TOKEN}',
    local_dir='/opt/damru/models'
)
print('Downloaded:', path)
"
fi
log "Model ready: /opt/damru/models/${DAMRU_MODEL_FILE}"

# ---- 4. llama.cpp server as systemd service (API-free local inference) ------
log "[4/8] Setting up llama.cpp server service..."
cat > /etc/systemd/system/damru-llama.service <<EOF
[Unit]
Description=Damru Local Inference Server (llama.cpp)
After=network.target
Restart=always

[Service]
Type=simple
User=ubuntu
EnvironmentFile=/etc/damru/secrets.env
ExecStart=/opt/llama.cpp/build/bin/llama-server \\
    -m /opt/damru/models/${DAMRU_MODEL_FILE} \\
    --host 0.0.0.0 \\
    --port ${LLAMA_PORT} \\
    -c 4096 \\
    -t $(nproc) \\
    --mlock \\
    -n 1024 \\
    --log-disable
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# ---- 5. SearXNG (self-hosted search -- NO Google API needed) ----------------
log "[5/8] Setting up SearXNG (free web search)..."
mkdir -p /opt/searxng
cat > /opt/searxng/docker-compose.yml <<'SEOF'
version: '3'
services:
  searxng:
    image: searxng/searxng:latest
    ports:
      - "8888:8080"
    volumes:
      - ./searxng:/etc/searxng
    environment:
      - SEARXNG_BASE_URL=http://localhost:8888
    restart: always
SEOF

mkdir -p /opt/searxng/searxng
cat > /opt/searxng/searxng/settings.yml <<'SEOF'
use_default_settings: true
search:
  safe_search: 0
  default_lang: "en"
server:
  secret_key: "damru-searxng-secret-key-change-this"
  limiter: false
SEOF

cd /opt/searxng && docker-compose up -d
log "SearXNG started on port 8888"

# ---- 6. Python venv + Curious Engine ----------------------------------------
log "[6/8] Installing Python stack + Curious Engine..."
mkdir -p /opt/damru
cd /opt/damru

if [ ! -d venv ]; then
    python3 -m venv venv
fi

/opt/damru/venv/bin/pip install -q --upgrade pip
/opt/damru/venv/bin/pip install -q \
    requests beautifulsoup4 feedparser newspaper3k \
    huggingface_hub datasets transformers \
    fastapi uvicorn schedule psutil \
    tiktoken sentencepiece

# Copy curious engine from repo
if [ ! -f /opt/damru/damru_curious_engine.py ]; then
    curl -s -H "Authorization: token ${HF_TOKEN}" \
        "https://raw.githubusercontent.com/Damru-AI/Damru-AI/main/damru_curious_engine.py" \
        -o /opt/damru/damru_curious_engine.py 2>/dev/null || \
    wget -q -O /opt/damru/damru_curious_engine.py \
        "https://raw.githubusercontent.com/Damru-AI/Damru-AI/main/damru_curious_engine.py"
fi

# ---- 7. Kaggle credentials --------------------------------------------------
log "[7/8] Setting up Kaggle credentials..."
mkdir -p /home/ubuntu/.kaggle /root/.kaggle
cat > /home/ubuntu/.kaggle/kaggle.json <<EOF
{"username":"${KAGGLE_USERNAME}","key":"${KAGGLE_KEY}"}
EOF
cp /home/ubuntu/.kaggle/kaggle.json /root/.kaggle/kaggle.json
chmod 600 /home/ubuntu/.kaggle/kaggle.json /root/.kaggle/kaggle.json
/opt/damru/venv/bin/pip install -q kaggle

# ---- 8. Systemd services + cron ---------------------------------------------
log "[8/8] Setting up auto-run services + cron..."

# Curious Engine service
cat > /etc/systemd/system/damru-curious.service <<EOF
[Unit]
Description=Damru Curious Learning Engine
After=network.target damru-llama.service
Requires=damru-llama.service

[Service]
Type=simple
User=ubuntu
EnvironmentFile=/etc/damru/secrets.env
WorkingDirectory=/opt/damru
ExecStart=/opt/damru/venv/bin/python3 /opt/damru/damru_curious_engine.py
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

# Enable all services
systemctl daemon-reload
systemctl enable damru-llama damru-curious
systemctl start damru-llama
sleep 5
systemctl start damru-curious

# Cron: Kaggle trigger every 11.5h, health check every 5 min
(crontab -l 2>/dev/null || true; cat <<'CRON'
# Damru auto-training trigger (every 11.5h)
0 */12 * * * /opt/damru/venv/bin/python3 /opt/damru/kaggle_trigger.py >> /var/log/damru-kaggle.log 2>&1
# Health check (every 5 min)
*/5 * * * * systemctl is-active damru-llama || systemctl restart damru-llama
*/5 * * * * systemctl is-active damru-curious || systemctl restart damru-curious
CRON
) | crontab -

# ---- Done -------------------------------------------------------------------
log ""
log "=================================================="
log " DAMRU SETUP COMPLETE! "
log "=================================================="
log " Local inference: http://localhost:${LLAMA_PORT}"
log " Search engine:  http://localhost:${SEARXNG_PORT}"
log " Curious Engine: running as service"
log " Auto-restart:   enabled (systemd)"
log " Kaggle trigger: cron (every 12h)"
log ""
log " Check status: systemctl status damru-llama damru-curious"
log " View logs:    journalctl -u damru-curious -f"
log "=================================================="
