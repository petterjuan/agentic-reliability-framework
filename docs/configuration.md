# Configuration Reference

**Agentic Reliability Framework (ARF) - Complete Configuration Guide**

---

## Table of Contents

- [Overview](#overview)
- [Environment Variables](#environment-variables)
- [Configuration File](#configuration-file)
- [Default Values](#default-values)
- [Advanced Configuration](#advanced-configuration)
- [Environment-Specific Configs](#environment-specific-configs)
- [Security Best Practices](#security-best-practices)

---

## Overview

ARF uses **environment variables** for all configuration, following the [12-Factor App](https://12factor.net/) methodology.

### Configuration Precedence

1. **Environment variables** (highest priority)
2. **`.env` file** (loaded via python-dotenv)
3. **Default values** in `config.py`

---

## Environment Variables

### API Configuration

#### `HF_API_KEY`

**Description:** Hugging Face API key for inference  
**Required:** Optional (falls back to local inference)  
**Default:** `""` (empty string)  
**Example:** `HF_API_KEY=hf_abc123def456`

**Where to get:**
1. Sign up at https://huggingface.co/
2. Go to Settings → Access Tokens
3. Create new token with `inference` scope

---

#### `HF_API_URL`

**Description:** Hugging Face Inference Router endpoint  
**Required:** No  
**Default:** `https://router.huggingface.co/hf-inference/v1/completions`  
**Example:** `HF_API_URL=https://custom-endpoint.com/v1/completions`

**When to change:**
- Using private HF inference endpoint
- Self-hosted inference server
- Custom model deployment

---

### System Configuration

#### `MAX_EVENTS_STORED`

**Description:** Maximum events retained in memory  
**Required:** No  
**Default:** `1000`  
**Range:** 100 - 10,000  
**Example:** `MAX_EVENTS_STORED=5000`

**Impact:**
- **Higher values:** More history, more memory usage
- **Lower values:** Less memory, less historical context

**Recommendation:**
- **Development:** 1000
- **Production (low traffic):** 5000
- **Production (high traffic):** 10000

---

#### `FAISS_BATCH_SIZE`

**Description:** Number of vectors to batch before FAISS write  
**Required:** No  
**Default:** `10`  
**Range:** 1 - 100  
**Example:** `FAISS_BATCH_SIZE=50`

**Impact:**
- **Higher values:** Fewer disk writes, more memory
- **Lower values:** More frequent saves, less memory

**Recommendation:**
- **SSD storage:** 50
- **HDD storage:** 10
- **Network storage:** 5

---

#### `FAISS_SAVE_INTERVAL_SECONDS`

**Description:** Minimum seconds between FAISS index saves  
**Required:** No  
**Default:** `30`  
**Range:** 10 - 300  
**Example:** `FAISS_SAVE_INTERVAL_SECONDS=60`

**Impact:**
- **Higher values:** Less I/O, higher data loss risk
- **Lower values:** More I/O, lower data loss risk

---

#### `VECTOR_DIM`

**Description:** Dimensionality of sentence embeddings  
**Required:** No  
**Default:** `384`  
**Valid values:** `384` (do not change unless changing model)

**Warning:** ⚠️ Changing this requires re-creating FAISS index

---

### Business Metrics

#### `BASE_REVENUE_PER_MINUTE`

**Description:** Baseline revenue per minute (USD)  
**Required:** No  
**Default:** `100.0`  
**Range:** 0.0 - ∞  
**Example:** `BASE_REVENUE_PER_MINUTE=1500.0`

**Used for:**
- Revenue loss calculations
- Business impact estimates
- ROI projections

**How to calculate:**
```
BASE_REVENUE_PER_MINUTE = Annual Revenue / (365 * 24 * 60)

Example:
$50M annual revenue = $50,000,000 / 525,600 = $95.13/minute
```

---

#### `BASE_USERS`

**Description:** Baseline active users  
**Required:** No  
**Default:** `1000`  
**Range:** 1 - ∞  
**Example:** `BASE_USERS=50000`

**Used for:**
- User impact calculations
- Capacity planning
- Scaling decisions

---

### Rate Limiting

#### `MAX_REQUESTS_PER_MINUTE`

**Description:** Maximum API requests per minute  
**Required:** No  
**Default:** `60`  
**Range:** 1 - 1000  
**Example:** `MAX_REQUESTS_PER_MINUTE=120`

**Impact:**
- Prevents API overload
- Controls costs
- Ensures fair usage

---

#### `MAX_REQUESTS_PER_HOUR`

**Description:** Maximum API requests per hour  
**Required:** No  
**Default:** `500`  
**Range:** 10 - 10,000  
**Example:** `MAX_REQUESTS_PER_HOUR=2000`

---

### Thresholds

#### `LATENCY_WARNING`

**Description:** Latency threshold for WARNING severity (ms)  
**Required:** No  
**Default:** `150.0`  
**Range:** 50.0 - 10,000.0  
**Example:** `LATENCY_WARNING=200.0`

---

#### `LATENCY_CRITICAL`

**Description:** Latency threshold for CRITICAL severity (ms)  
**Required:** No  
**Default:** `300.0`  
**Range:** 100.0 - 10,000.0  
**Example:** `LATENCY_CRITICAL=500.0`

---

#### `LATENCY_EXTREME`

**Description:** Latency threshold for EXTREME severity (ms)  
**Required:** No  
**Default:** `500.0`  
**Range:** 200.0 - 10,000.0  
**Example:** `LATENCY_EXTREME=1000.0`

---

### Logging

#### `LOG_LEVEL`

**Description:** Application logging level  
**Required:** No  
**Default:** `INFO`  
**Valid values:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`  
**Example:** `LOG_LEVEL=DEBUG`

**When to use:**
- **DEBUG:** Development, troubleshooting
- **INFO:** Production (default)
- **WARNING:** Production (quiet)
- **ERROR:** Critical systems only

---

## Configuration File

### config.py

```python
"""
Configuration management for ARF
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load .env file if exists
load_dotenv()


@dataclass
class AppConfig:
    """Application configuration from environment variables"""
    
    # API Configuration
    hf_api_key: str = os.getenv("HF_API_KEY", "")
    hf_api_url: str = os.getenv(
        "HF_API_URL", 
        "https://router.huggingface.co/hf-inference/v1/completions"
    )
    
    # System Configuration
    max_events_stored: int = int(os.getenv("MAX_EVENTS_STORED", "1000"))
    faiss_batch_size: int = int(os.getenv("FAISS_BATCH_SIZE", "10"))
    faiss_save_interval: int = int(os.getenv("FAISS_SAVE_INTERVAL_SECONDS", "30"))
    vector_dim: int = int(os.getenv("VECTOR_DIM", "384"))
    
    # Business Metrics
    base_revenue_per_minute: float = float(os.getenv("BASE_REVENUE_PER_MINUTE", "100.0"))
    base_users: int = int(os.getenv("BASE_USERS", "1000"))
    
    # Rate Limiting
    max_requests_per_minute: int = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))
    max_requests_per_hour: int = int(os.getenv("MAX_REQUESTS_PER_HOUR", "500"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Thresholds
    latency_warning: float = float(os.getenv("LATENCY_WARNING", "150.0"))
    latency_critical: float = float(os.getenv("LATENCY_CRITICAL", "300.0"))
    latency_extreme: float = float(os.getenv("LATENCY_EXTREME", "500.0"))
    
    # File Paths
    index_file: str = os.getenv("INDEX_FILE", "data/faiss_index.bin")
    texts_file: str = os.getenv("TEXTS_FILE", "data/incident_texts.json")
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables"""
        return cls()
    
    def validate(self) -> bool:
        """Validate configuration"""
        if self.vector_dim <= 0:
            raise ValueError("VECTOR_DIM must be positive")
        if self.max_events_stored <= 0:
            raise ValueError("MAX_EVENTS_STORED must be positive")
        if self.faiss_batch_size <= 0:
            raise ValueError("FAISS_BATCH_SIZE must be positive")
        return True


# Global config instance
config = AppConfig.from_env()
config.validate()
```

---

## Default Values

### Complete Default Configuration

```bash
# API Configuration
HF_API_KEY=""
HF_API_URL="https://router.huggingface.co/hf-inference/v1/completions"

# System Configuration
MAX_EVENTS_STORED=1000
FAISS_BATCH_SIZE=10
FAISS_SAVE_INTERVAL_SECONDS=30
VECTOR_DIM=384

# Business Metrics
BASE_REVENUE_PER_MINUTE=100.0
BASE_USERS=1000

# Rate Limiting
MAX_REQUESTS_PER_MINUTE=60
MAX_REQUESTS_PER_HOUR=500

# Logging
LOG_LEVEL=INFO

# Thresholds
LATENCY_WARNING=150.0
LATENCY_CRITICAL=300.0
LATENCY_EXTREME=500.0

# File Paths
INDEX_FILE=data/faiss_index.bin
TEXTS_FILE=data/incident_texts.json
```

---

## Advanced Configuration

### Custom File Paths

```bash
# Store data in custom location
INDEX_FILE=/mnt/storage/arf/faiss_index.bin
TEXTS_FILE=/mnt/storage/arf/incident_texts.json
```

### Multiple Environments

```bash
# .env.development
LOG_LEVEL=DEBUG
MAX_EVENTS_STORED=500
BASE_REVENUE_PER_MINUTE=10.0

# .env.staging
LOG_LEVEL=INFO
MAX_EVENTS_STORED=2000
BASE_REVENUE_PER_MINUTE=50.0

# .env.production
LOG_LEVEL=WARNING
MAX_EVENTS_STORED=10000
BASE_REVENUE_PER_MINUTE=1500.0
```

**Load specific environment:**

```bash
# Development
cp .env.development .env
python app.py

# Production
cp .env.production .env
python app.py
```

---

### Performance Tuning

#### High-Traffic Configuration

```bash
# Optimized for high event volume
MAX_EVENTS_STORED=10000
FAISS_BATCH_SIZE=100
FAISS_SAVE_INTERVAL_SECONDS=60
MAX_REQUESTS_PER_MINUTE=500
MAX_REQUESTS_PER_HOUR=10000
```

#### Low-Memory Configuration

```bash
# Optimized for limited memory
MAX_EVENTS_STORED=500
FAISS_BATCH_SIZE=5
FAISS_SAVE_INTERVAL_SECONDS=15
```

#### Fast-Recovery Configuration

```bash
# Optimized for minimal data loss
FAISS_BATCH_SIZE=1
FAISS_SAVE_INTERVAL_SECONDS=10
```

---

## Environment-Specific Configs

### Development

```bash
# .env.development
HF_API_KEY=hf_dev_token
LOG_LEVEL=DEBUG
MAX_EVENTS_STORED=100
BASE_REVENUE_PER_MINUTE=1.0
BASE_USERS=10
```

### Staging

```bash
# .env.staging
HF_API_KEY=hf_staging_token
LOG_LEVEL=INFO
MAX_EVENTS_STORED=2000
BASE_REVENUE_PER_MINUTE=100.0
BASE_USERS=1000
```

### Production

```bash
# .env.production
HF_API_KEY=hf_prod_token
LOG_LEVEL=WARNING
MAX_EVENTS_STORED=10000
BASE_REVENUE_PER_MINUTE=2500.0
BASE_USERS=100000
MAX_REQUESTS_PER_MINUTE=1000
MAX_REQUESTS_PER_HOUR=50000
```

---

## Security Best Practices

### API Keys

❌ **NEVER commit .env to git:**

```bash
# Add to .gitignore
echo ".env" >> .gitignore
echo ".env.*" >> .gitignore
```

✅ **Use secrets management:**

```bash
# AWS Secrets Manager
aws secretsmanager create-secret \
    --name arf/hf-api-key \
    --secret-string "hf_abc123"

# Fetch in application
HF_API_KEY=$(aws secretsmanager get-secret-value \
    --secret-id arf/hf-api-key \
    --query SecretString \
    --output text)
```

✅ **Use environment-specific keys:**

```bash
# Development
HF_API_KEY=hf_dev_readonly_token

# Production
HF_API_KEY=hf_prod_with_limits_token
```

---

### File Permissions

```bash
# Restrict .env file permissions
chmod 600 .env

# Owner read/write only
-rw------- 1 user user .env
```

---

### Docker Secrets

```bash
# Create Docker secret
echo "hf_abc123" | docker secret create hf_api_key -

# Use in docker-compose.yml
services:
  arf:
    secrets:
      - hf_api_key
    environment:
      - HF_API_KEY=/run/secrets/hf_api_key
```

---

## Configuration Validation

### Startup Validation

ARF validates configuration on startup:

```python
# config.py validates:
- Positive values for counts/sizes
- Valid log levels
- File path accessibility
- API key format (if provided)
```

**Example validation error:**

```
ValueError: MAX_EVENTS_STORED must be positive (got: -100)
```

---

### Runtime Validation

```python
from config import config

# Access validated config
print(f"Max events: {config.max_events_stored}")
print(f"Revenue/min: ${config.base_revenue_per_minute}")
```

---

## Troubleshooting

### Config Not Loading

```bash
# Check .env file exists
ls -la .env

# Verify python-dotenv installed
pip show python-dotenv

# Test loading
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('HF_API_KEY'))"
```

---

### Environment Variable Not Working

```bash
# Check if set
echo $HF_API_KEY

# Set temporarily
export HF_API_KEY=hf_abc123
python app.py

# Set permanently (Linux)
echo "export HF_API_KEY=hf_abc123" >> ~/.bashrc
source ~/.bashrc
```

---

### Docker Environment Variables

```bash
# Pass via docker run
docker run -e HF_API_KEY=hf_abc123 arf:latest

# Pass via env file
docker run --env-file .env arf:latest

# Check inside container
docker exec arf-container env | grep HF_API_KEY
```

---

## Configuration Examples

### E-commerce Platform

```bash
# High traffic, high revenue
MAX_EVENTS_STORED=10000
BASE_REVENUE_PER_MINUTE=5000.0
BASE_USERS=500000
LATENCY_WARNING=100.0
LATENCY_CRITICAL=200.0
MAX_REQUESTS_PER_MINUTE=2000
```

---

### SaaS Application

```bash
# Medium traffic, predictable load
MAX_EVENTS_STORED=5000
BASE_REVENUE_PER_MINUTE=500.0
BASE_USERS=50000
LATENCY_WARNING=150.0
LATENCY_CRITICAL=300.0
MAX_REQUESTS_PER_MINUTE=500
```

---

### Internal Tool

```bash
# Low traffic, development use
MAX_EVENTS_STORED=1000
BASE_REVENUE_PER_MINUTE=0.0
BASE_USERS=100
LATENCY_WARNING=500.0
LATENCY_CRITICAL=1000.0
MAX_REQUESTS_PER_MINUTE=60
LOG_LEVEL=DEBUG
```

---

## Migration Guide

### Upgrading from v1.0 to v2.0

**New variables:**
- `FAISS_SAVE_INTERVAL_SECONDS` (replaces hardcoded value)
- `INDEX_FILE` and `TEXTS_FILE` (customizable paths)

**Deprecated:**
- None (all configs backward compatible)

**Action required:**
```bash
# Add new variables to .env
echo "FAISS_SAVE_INTERVAL_SECONDS=30" >> .env
echo "INDEX_FILE=data/faiss_index.bin" >> .env
echo "TEXTS_FILE=data/incident_texts.json" >> .env
```

---

## Professional Configuration Support

**Need help optimizing ARF for your infrastructure?**

LGCY Labs offers:
- ✅ Custom configuration tuning
- ✅ Performance optimization
- ✅ Security hardening
- ✅ Multi-environment setup
- ✅ Monitoring integration

**[📅 Book Consultation](https://calendly.com/petter2025us/30min)** • **[💼 Professional Services](https://lgcylabs.vercel.app/)**

---

**Built by [LGCY Labs](https://lgcylabs.vercel.app/) | [Back to README](../README.md)**
