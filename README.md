# ═══════════════════════════════════════════════════════════════════════════════
#  Ollama Intelligent Model Scheduler — Deployment Guide
#  Target: AWS g5.2xlarge · Ubuntu 22.04 · Ollama · Python 3.11+
# ═══════════════════════════════════════════════════════════════════════════════

## 1. Project layout

    /opt/model_scheduler/
    ├── config.py           ← all tuneable constants (env-var overridable)
    ├── schemas.py          ← PendingRequest, LatencyRecord dataclasses
    ├── ollama_client.py    ← async httpx wrapper for Ollama REST API
    ├── queue_manager.py    ← RequestQueue with model-affinity batch reordering
    ├── scheduler.py        ← ModelScheduler worker + LatencyTracker
    ├── main.py             ← FastAPI app + /api/generate, /api/chat, /health …
    ├── requirements.txt
    └── nginx.conf          ← drop into /etc/nginx/sites-available/


## 2. Install

    sudo apt-get install -y python3.11 python3.11-venv
    python3.11 -m venv /opt/model_scheduler/venv
    source /opt/model_scheduler/venv/bin/activate
    pip install -r /opt/model_scheduler/requirements.txt


## 3. Environment overrides  (create /opt/model_scheduler/.env)

    SCHEDULER_BATCH_WINDOW_SIZE=10
    SCHEDULER_BATCH_COLLECT_TIMEOUT_S=0.20
    SCHEDULER_KEEP_ALIVE_ACTIVE=5m
    SCHEDULER_VRAM_BUDGET_GB=22.0
    SCHEDULER_MAX_QUEUE_SIZE=500
    SCHEDULER_REQUEST_TIMEOUT_S=600
    SCHEDULER_LOG_LEVEL=INFO
    SCHEDULER_PORT=8080


## 4. Systemd service  (/etc/systemd/system/ollama-scheduler.service)

    [Unit]
    Description=Ollama Intelligent Model Scheduler
    After=network.target ollama.service
    Requires=ollama.service

    [Service]
    Type=simple
    User=ubuntu
    WorkingDirectory=/opt/model_scheduler
    ExecStart=/opt/model_scheduler/venv/bin/python main.py
    Restart=always
    RestartSec=3
    StandardOutput=journal
    StandardError=journal
    Environment=PYTHONUNBUFFERED=1

    # Graceful shutdown — give the worker time to drain in-flight requests
    TimeoutStopSec=30
    KillSignal=SIGINT

    [Install]
    WantedBy=multi-user.target

    # Enable:
    #   sudo systemctl daemon-reload
    #   sudo systemctl enable --now ollama-scheduler


## 5. Nginx setup

    sudo cp /opt/model_scheduler/nginx.conf \
            /etc/nginx/sites-available/ollama-scheduler
    sudo ln -s /etc/nginx/sites-available/ollama-scheduler \
               /etc/nginx/sites-enabled/ollama-scheduler
    sudo nginx -t && sudo systemctl reload nginx


## 6. EBS optimisation (gp3, ≥ 3 000 IOPS)

    # Check current IOPS
    aws ec2 describe-volumes --filters Name=attachment.instance-id,Values=$(ec2-metadata -i | cut -d' ' -f2)

    # Modify to gp3 with 6 000 IOPS + 250 MB/s throughput (no charge above 3 000 IOPS is free for gp3)
    aws ec2 modify-volume \
      --volume-id vol-XXXXXXXX \
      --volume-type gp3 \
      --iops 6000 \
      --throughput 250


## 7. Quick smoke test

    # Non-streaming (blocks until complete)
    curl -s http://localhost/api/generate \
      -d '{"model":"llama3.2:3b","prompt":"Hello!","stream":false}' | jq .

    # Streaming
    curl -N http://localhost/api/generate \
      -d '{"model":"llama3.2:3b","prompt":"Count to 5","stream":true}'

    # Scheduler health
    curl -s http://localhost/health | jq .

    # Per-model latency metrics
    curl -s "http://localhost/metrics?model=llama3.2:3b" | jq .

    # Live queue snapshot
    curl -s http://localhost/queue/status | jq .


## 8. Key architectural decisions

    Batch-window affinity scheduling
    ──────────────────────────────────
    get_batch() pulls up to BATCH_WINDOW_SIZE (10) items and partitions them:
      • requests for the currently-loaded model → front of batch
      • requests for other models               → back of batch
    This guarantees the GPU drains all 2B work before paying the cost of
    loading a 26B model.  On a busy instance with mixed traffic, this can
    reduce VRAM swaps by 80–90 %.

    Pure execution latency
    ──────────────────────
    execution_start_time is stamped immediately before the first byte is sent
    to Ollama.  queue_wait_ms is logged separately.  This separation lets you
    distinguish scheduler contention from true model throughput degradation.

    VRAM-aware model switching
    ──────────────────────────
    Before switching models, _handle_model_switch() checks:
      vram(current) + vram(new) > VRAM_BUDGET_GB  (default 22 GB on A10G)
    If true → send keep_alive=0 to Ollama to force-evict the current model,
    then sleep 1 s for the CUDA allocator to reclaim pages, then proceed.
    If false → let Ollama's own LRU eviction handle it (saves time for
    cheap swaps like 2B → 4B).

    Single worker guarantee
    ───────────────────────
    Only ONE asyncio Task drives the scheduler loop.  This makes VRAM state
    fully deterministic: there is never a race between two requests trying
    to load different models simultaneously.  For higher throughput on
    multi-GPU instances, you would shard the queue by model family instead.
