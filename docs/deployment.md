# Deployment Guide

**Agentic Reliability Framework (ARF) - Production Deployment**

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Platforms](#cloud-platforms)
- [Production Checklist](#production-checklist)
- [Monitoring & Observability](#monitoring--observability)
- [Troubleshooting](#troubleshooting)
- [Scaling Strategies](#scaling-strategies)

---

## Prerequisites

### System Requirements

**Minimum:**
- 2 CPU cores
- 4 GB RAM
- 10 GB disk space
- Python 3.12+

**Recommended:**
- 4 CPU cores
- 8 GB RAM
- 50 GB disk space (for FAISS index growth)
- Python 3.12+

### Dependencies

```bash
# System dependencies
sudo apt-get update
sudo apt-get install -y \
    python3.12 \
    python3-pip \
    git \
    build-essential

# Python dependencies (handled by pip)
```

---

## Local Development

### 1. Clone Repository

```bash
git clone https://github.com/petterjuan/agentic-reliability-framework.git
cd agentic-reliability-framework
```

### 2. Create Virtual Environment

```bash
# Create venv
python3.12 -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit configuration
nano .env
```

**Minimum required:**
```bash
HF_API_KEY=your_huggingface_api_key_here
```

### 5. Run Locally

```bash
python app.py
```

Access at: http://localhost:7860

---

## Docker Deployment

### Option 1: Pre-built Image (Coming Soon)

```bash
docker pull lgcylabs/arf:latest
docker run -p 7860:7860 --env-file .env lgcylabs/arf:latest
```

### Option 2: Build from Source

#### **Dockerfile**

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directories
RUN mkdir -p data

# Expose Gradio port
EXPOSE 7860

# Set environment variables
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import requests; requests.get('http://localhost:7860')"

# Run application
CMD ["python", "app.py"]
```

#### **Build Image**

```bash
docker build -t arf:latest .
```

#### **Run Container**

```bash
docker run -d \
    --name arf-production \
    -p 7860:7860 \
    --env-file .env \
    -v $(pwd)/data:/app/data \
    --restart unless-stopped \
    arf:latest
```

#### **View Logs**

```bash
docker logs -f arf-production
```

---

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  arf:
    build: .
    container_name: arf-production
    ports:
      - "7860:7860"
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:7860')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

**Deploy:**

```bash
docker-compose up -d
```

---

## Cloud Platforms

### AWS Deployment

#### **EC2 Instance**

```bash
# 1. Launch EC2 instance (t3.medium or larger)
# - Ubuntu 22.04 LTS
# - 8 GB RAM, 2 vCPUs
# - Security group: Allow port 7860

# 2. SSH into instance
ssh -i key.pem ubuntu@<instance-ip>

# 3. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# 4. Clone repo
git clone https://github.com/petterjuan/agentic-reliability-framework.git
cd agentic-reliability-framework

# 5. Configure
cp .env.example .env
nano .env

# 6. Deploy
docker-compose up -d
```

#### **AWS ECS (Elastic Container Service)**

1. **Build and push to ECR:**

```bash
# Create ECR repository
aws ecr create-repository --repository-name arf

# Authenticate Docker
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -t arf:latest .
docker tag arf:latest <account>.dkr.ecr.us-east-1.amazonaws.com/arf:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/arf:latest
```

2. **Create ECS Task Definition:**

```json
{
  "family": "arf-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "arf",
      "image": "<account>.dkr.ecr.us-east-1.amazonaws.com/arf:latest",
      "portMappings": [
        {
          "containerPort": 7860,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "HF_API_KEY", "value": "your_key"},
        {"name": "LOG_LEVEL", "value": "INFO"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/arf",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "arf"
        }
      }
    }
  ]
}
```

3. **Create ECS Service:**

```bash
aws ecs create-service \
    --cluster arf-cluster \
    --service-name arf-service \
    --task-definition arf-task \
    --desired-count 2 \
    --launch-type FARGATE \
    --load-balancer targetGroupArn=<arn>,containerName=arf,containerPort=7860
```

---

### Google Cloud Platform (GCP)

#### **Google Cloud Run**

```bash
# 1. Build and push to GCR
gcloud builds submit --tag gcr.io/<project-id>/arf

# 2. Deploy to Cloud Run
gcloud run deploy arf \
    --image gcr.io/<project-id>/arf \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 7860 \
    --memory 2Gi \
    --cpu 2 \
    --set-env-vars HF_API_KEY=your_key
```

#### **GCP Compute Engine**

```bash
# 1. Create instance
gcloud compute instances create arf-instance \
    --machine-type=n1-standard-2 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB

# 2. SSH and deploy
gcloud compute ssh arf-instance
# Follow EC2 deployment steps
```

---

### Azure

#### **Azure Container Instances**

```bash
# 1. Build and push to ACR
az acr build --registry <registry-name> --image arf:latest .

# 2. Deploy container instance
az container create \
    --resource-group arf-rg \
    --name arf-instance \
    --image <registry-name>.azurecr.io/arf:latest \
    --cpu 2 \
    --memory 4 \
    --ports 7860 \
    --environment-variables HF_API_KEY=your_key
```

---

### Hugging Face Spaces

#### **Deploy to HF Spaces**

1. **Create Space:**
   - Go to https://huggingface.co/new-space
   - Select "Gradio" SDK
   - Name: `agentic-reliability-framework`

2. **Push Code:**

```bash
# Clone space
git clone https://huggingface.co/spaces/<username>/agentic-reliability-framework
cd agentic-reliability-framework

# Copy files
cp /path/to/arf/app.py .
cp /path/to/arf/requirements.txt .
cp -r /path/to/arf/models.py healing_policies.py config.py .

# Commit
git add .
git commit -m "Initial deployment"
git push
```

3. **Configure Secrets:**
   - Go to Space settings → Variables & Secrets
   - Add `HF_API_KEY`

**Space will automatically deploy!**

---

### Railway

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Initialize project
railway init

# 4. Deploy
railway up
```

---

## Production Checklist

### Security

- [ ] Set strong `HF_API_KEY`
- [ ] Enable HTTPS/TLS
- [ ] Configure firewall rules
- [ ] Set up authentication (if needed)
- [ ] Regular security updates
- [ ] Audit logs enabled

### Performance

- [ ] Configure `MAX_EVENTS_STORED` appropriately
- [ ] Tune `FAISS_BATCH_SIZE` for your load
- [ ] Enable lazy-loading (default)
- [ ] Set appropriate `LOG_LEVEL` (INFO in production)
- [ ] Configure rate limits

### Reliability

- [ ] Set up health checks
- [ ] Configure auto-restart on failure
- [ ] Enable persistent volumes for data
- [ ] Configure backup strategy
- [ ] Set up monitoring & alerts
- [ ] Document recovery procedures

### Monitoring

- [ ] Application logs
- [ ] Performance metrics
- [ ] Error tracking
- [ ] Resource utilization
- [ ] Business metrics dashboard

---

## Monitoring & Observability

### Application Logs

```bash
# Docker
docker logs -f arf-production

# systemd service
journalctl -u arf -f

# File-based
tail -f logs/arf.log
```

### Health Check Endpoint

```python
# Add to app.py
@app.route("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }
```

**Monitor:**
```bash
curl http://localhost:7860/health
```

---

### Prometheus Metrics (Optional)

Install `prometheus-client`:

```bash
pip install prometheus-client
```

Add to `app.py`:

```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
events_processed = Counter('arf_events_processed_total', 'Total events processed')
analysis_duration = Histogram('arf_analysis_duration_seconds', 'Analysis duration')

@app.route("/metrics")
def metrics():
    return generate_latest()
```

---

### Grafana Dashboard

Connect Prometheus to Grafana and import dashboard:

**Key Metrics:**
- Events processed per minute
- Average analysis latency
- Agent confidence scores
- Policy trigger rate
- Business impact (revenue saved)

---

## Troubleshooting

### Common Issues

#### **Port Already in Use**

```bash
# Find process using port 7860
lsof -i :7860

# Kill process
kill -9 <PID>
```

#### **FAISS Index Errors**

```bash
# Delete corrupted index
rm data/faiss_index.bin
rm data/incident_texts.json

# Restart (will create new index)
```

#### **Out of Memory**

```bash
# Reduce max events
echo "MAX_EVENTS_STORED=500" >> .env

# Or increase container memory
docker update --memory 4g arf-production
```

#### **Slow Startup**

```bash
# Confirm lazy-loading is enabled
grep "model = None" app.py

# Should see: model = None  # Lazy-loaded
```

---

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python app.py
```

---

## Scaling Strategies

### Vertical Scaling

**Increase resources:**

```bash
# Docker
docker update --cpus 4 --memory 8g arf-production

# Kubernetes
kubectl scale deployment arf --replicas=1
kubectl set resources deployment arf -c=arf --limits=cpu=4,memory=8Gi
```

---

### Horizontal Scaling

**Load Balancer + Multiple Instances:**

```
                    ┌─────────┐
                    │   LB    │
                    └─┬─┬─┬─┬─┘
                      │ │ │ │
           ┌──────────┘ │ │ └──────────┐
           │            │ │            │
           ▼            ▼ ▼            ▼
     ┌─────────┐  ┌─────────┐  ┌─────────┐
     │ ARF #1  │  │ ARF #2  │  │ ARF #3  │
     └─────────┘  └─────────┘  └─────────┘
           │            │            │
           └────────────┴────────────┘
                        │
                   ┌────▼────┐
                   │ Shared  │
                   │  Data   │
                   └─────────┘
```

**Docker Compose:**

```yaml
version: '3.8'

services:
  arf:
    build: .
    deploy:
      replicas: 3
    ports:
      - "7860-7862:7860"
    volumes:
      - shared-data:/app/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - arf

volumes:
  shared-data:
```

---

### Database-Backed Storage

**For multi-instance deployments:**

Replace `ThreadSafeEventStore` with PostgreSQL:

```python
# Use SQLAlchemy for event persistence
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker

engine = create_engine(os.getenv("DATABASE_URL"))
Session = sessionmaker(bind=engine)

class DatabaseEventStore:
    def add(self, event):
        session = Session()
        session.add(event)
        session.commit()
```

---

## Backup & Recovery

### Data Backup

```bash
# Backup FAISS index + texts
tar -czf arf-backup-$(date +%Y%m%d).tar.gz data/

# Upload to S3
aws s3 cp arf-backup-*.tar.gz s3://arf-backups/
```

### Automated Backups

```bash
# Add to crontab
0 2 * * * /path/to/backup-script.sh
```

**backup-script.sh:**
```bash
#!/bin/bash
DATE=$(date +%Y%m%d)
tar -czf /backups/arf-$DATE.tar.gz /app/data
find /backups -name "arf-*.tar.gz" -mtime +30 -delete
```

---

### Disaster Recovery

```bash
# 1. Stop application
docker stop arf-production

# 2. Restore backup
tar -xzf arf-backup-20251209.tar.gz -C /app/

# 3. Restart
docker start arf-production
```

---

## Cost Optimization

### Cloud Provider Costs

**AWS:**
- t3.medium EC2: ~$30/month
- ECS Fargate (2 tasks): ~$50/month

**GCP:**
- Cloud Run: $0 (if <2M requests/month)
- e2-medium Compute: ~$25/month

**Azure:**
- B2s VM: ~$30/month

**Hugging Face Spaces:**
- Free tier available
- Paid: $5-$50/month

---

## Professional Deployment Support

**Need help deploying ARF in your infrastructure?**

LGCY Labs offers:
- ✅ Custom deployment (AWS/GCP/Azure)
- ✅ High-availability setup
- ✅ Monitoring & alerting
- ✅ Team training
- ✅ 3 months support

**[📅 Book Consultation](https://calendly.com/petter2025us/30min)** • **[💼 Professional Services](https://lgcylabs.vercel.app/)**

---

**Built by [LGCY Labs](https://lgcylabs.vercel.app/) | [Back to README](../README.md)**
