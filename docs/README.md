<p align="center">
  <img src="https://dummyimage.com/1200x260/000/fff&text=AGENTIC+RELIABILITY+FRAMEWORK" width="100%" alt="Agentic Reliability Framework Banner" />
</p>

<h2 align="center"><p align="center">
  Enterprise-Grade Multi-Agent AI for Autonomous System Reliability & Self-Healing
</p></h2>

> **Fortune 500-grade AI system for production reliability monitoring**  
> Built by engineers who managed $1M+ incidents at scale

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/agentic-reliability-framework?style=for-the-badge)](https://pypi.org/project/agentic-reliability-framework/)
[![Python Versions](https://img.shields.io/pypi/pyversions/agentic-reliability-framework?style=for-the-badge)](https://pypi.org/project/agentic-reliability-framework/)
[![Tests](https://img.shields.io/badge/tests-157%2F158%20passing-brightgreen?style=for-the-badge)](https://github.com/petterjuan/agentic-reliability-framework/actions)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](./LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-Live%20Demo-yellow?style=for-the-badge)](https://huggingface.co/spaces/petter2025/agentic-reliability-framework)

**[🚀 Try Live Demo](https://huggingface.co/spaces/petter2025/agentic-reliability-framework)** • **[📚 Documentation](https://github.com/petterjuan/agentic-reliability-framework/tree/main/docs)** • **[💼 Get Professional Help](https://lgcylabs.vercel.app/)**

</div>

# Agentic Reliability Framework (ARF) v3.3.0

**Production-grade multi-agent AI for reliability monitoring and advisory recommendations**

[![PyPI version](https://img.shields.io/pypi/v/agentic-reliability-framework.svg)](https://pypi.org/project/agentic-reliability-framework/)
[![Python Versions](https://img.shields.io/pypi/pyversions/agentic-reliability-framework.svg)](https://pypi.org/project/agentic-reliability-framework/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## 🎯 What is ARF?

The Agentic Reliability Framework (ARF) is an open-source AI system that:
- **Monitors** your production systems for reliability issues
- **Analyzes** incidents using RAG (Retrieval-Augmented Generation)
- **Recommends** healing actions based on historical data
- **Provides** advisory insights without automatic execution

## ✨ Key Features (OSS Edition)

### 🔍 **Incident Analysis**
- Real-time anomaly detection with multiple algorithms
- Business impact calculation and prioritization
- RAG-powered historical similarity search
- Multi-agent collaboration for root cause analysis

### 🧠 **AI-Powered Recommendations**
- Creates **HealingIntent** recommendations
- Analyzes 1000+ historical incidents for context
- Confidence scoring based on historical success rates
- Safety-first approach with policy enforcement

### 📊 **Memory & Learning**
- In-memory RAG graph with FAISS similarity search
- Stores incident patterns and successful resolutions
- 7-day lookback window for historical context
- Custom embedding model for your domain

### ⚙️ **Advisory MCP Server**
- **Advisory mode only** (Enterprise adds approval/autonomous)
- Tool analysis without execution
- Safety validation and policy checks
- Integration-ready API

## 🚀 Quick Start

```bash
# Install the OSS framework
pip install agentic-reliability-framework

# Start with default configuration
python -m agentic_reliability_framework.cli --demo
```

## Basic Usage
```
from agentic_reliability_framework import create_mcp_client

# Create an advisory MCP client
client = create_mcp_client()

# Analyze a potential issue
analysis = await client.execute_tool({
    "tool": "restart_container",
    "component": "api-service",
    "parameters": {"delay": 30},
    "justification": "High latency spikes detected"
})

print(f"Recommendation: {analysis['result']['message']}")
print(f"Requires Enterprise: {analysis['result']['requires_enterprise']}")
```

## 📁 Project Structure

```
agentic_reliability_framework/
├── engine/           # Core engines (reliability, anomaly, MCP)
├── memory/           # RAG graph and FAISS storage
├── models/           # Data models (HealingIntent, etc.)
├── config.py         # Configuration management
├── healing_policies.py # Safety policies
└── lazy.py          # Lazy loading for performance
```

## 🔗 OSS vs Enterprise

FeatureOSS EditionEnterprise Edition**MCP Modes**Advisory onlyAdvisory, Approval, Autonomous**Storage**In-memory (1000 incidents)Persistent (unlimited)**Execution**Analysis onlyFull execution with audit**Learning**Historical lookupContinuous learning**Audit Trails**NoneComprehensive compliance**License**Apache 2.0Commercial

**Upgrade to Enterprise for:**

*   Autonomous execution of healing actions
    
*   Approval workflows with human oversight
    
*   Persistent storage and unlimited history
    
*   Learning engine that improves over time
    
*   Compliance-ready audit trails
    

[Get Enterprise License →](https://arf.dev/enterprise)

## 🧪 Testing

```
# Run all tests
pytest tests/

# Run specific test suites
pytest tests/test_models.py -v
pytest tests/test_mcp_integration.py -v
```
## 🤝 Contributing
We welcome contributions! Please see [CONTRIBUTING.md](https://contributing.md/) for guidelines.

1.  Fork the repository
    
2.  Create a feature branch
    
3.  Make your changes
    
4.  Add tests
    
5.  Submit a pull request
    

📄 License
----------

This project is licensed under the **Apache License 2.0** - see the [LICENSE](https://license/) file for details.

The OSS edition is free forever. For commercial use with execution capabilities, see [Enterprise Edition](https://github.com/petterjuan/agentic-reliability-enterprise).

📞 Support
----------

*   **OSS Issues**: [GitHub Issues](https://github.com/petterjuan/agentic-reliability-framework/issues)
    
*   **Enterprise Support**: [Contact Sales](https://arf.dev/contact)
    
*   **Documentation**: [Read the Docs](https://arf.dev/docs)
    

**Maintained by** [**ARF Labs**](https://arf.dev/) · **Version 3.3.0** · **Apache 2.0 Licensed**
