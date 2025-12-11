"""
Command-line interface for ARF
"""

import click
import sys
from arf.__version__ import __version__
from arf.app import create_enhanced_ui

@click.group()
@click.version_option(version=__version__)
def main():
    """Agentic Reliability Framework - Multi-Agent AI for Production Reliability"""
    pass

@main.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=7860, type=int, help='Port to bind to')
@click.option('--share/--no-share', default=False, help='Create public Gradio share link')
def serve(host, port, share):
    """Start the ARF Gradio UI server"""
    click.echo(f"🚀 Starting ARF v{__version__} on {host}:{port}...")
    demo = create_enhanced_ui()
    demo.launch(server_name=host, server_port=port, share=share)

@main.command()
def version():
    """Show ARF version"""
    click.echo(f"Agentic Reliability Framework v{__version__}")

@main.command()
def doctor():
    """Check ARF installation and dependencies"""
    click.echo("🔍 Checking ARF installation...")
    
    dependencies = [
        ("FAISS", "faiss"),
        ("SentenceTransformers", "sentence_transformers"),
        ("Gradio", "gradio"),
        ("Pandas", "pandas"),
        ("NumPy", "numpy"),
        ("Pydantic", "pydantic"),
        ("Requests", "requests"),
        ("CircuitBreaker", "circuitbreaker"),
        ("atomicwrites", "atomicwrites"),
        ("python-dotenv", "dotenv"),
        ("Click", "click")
    ]
    
    all_ok = True
    for name, module in dependencies:
        try:
            __import__(module.replace("-", "_"))
            click.echo(f"  ✓ {name}")
        except ImportError:
            click.echo(f"  ✗ {name} missing", err=True)
            all_ok = False
    
    if all_ok:
        click.echo("\n✅ All dependencies OK!")
    else:
        click.echo("\n❌ Missing dependencies. Install with: pip install agentic-reliability-framework", err=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
