import os
from setuptools import setup, find_packages

# Read README if it exists
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="agentic-reliability-framework",
    version="3.3.0",
    packages=find_packages(),
    install_requires=[
        "gradio>=4.19.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "faiss-cpu>=1.7.0",
        "requests>=2.31.0",
        "circuitbreaker>=1.4.0",
        "atomicwrites>=1.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "mypy>=1.0.0",
            "black>=23.0.0",
            "types-requests",
            "types-PyYAML",
        ]
    },
    python_requires=">=3.10",
    description="Agentic Reliability Framework - OSS Edition: AI-powered infrastructure reliability monitoring",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Juan Petter",
    author_email="your-email@example.com",
    url="https://github.com/petterjuan/agentic-reliability-framework",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: System :: Monitoring",
        "Topic :: System :: Networking :: Monitoring",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="reliability, monitoring, ai, agents, infrastructure, devops",
)
