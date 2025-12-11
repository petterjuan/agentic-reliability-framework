"""
Fast version command for CLI - doesn't load FAISS
"""

import click
import sys

# Try to get version without importing arf
try:
    # Direct file read to avoid imports
    import os
    version_file = os.path.join(os.path.dirname(__file__), '__version__.py')
    with open(version_file, 'r') as f:
        for line in f:
            if '__version__' in line:
                version = line.split('=')[1].strip().strip("'\"")
                break
    VERSION = version
except:
    VERSION = "unknown"

@click.group()
@click.version_option(version=VERSION)  # Use pre-loaded version
def main():
    """Agentic Reliability Framework - Multi-Agent AI for Production Reliability"""
    pass

@main.command()
def version():
    """Show ARF version (FAST - no FAISS load)"""
    click.echo(f"Agentic Reliability Framework v{VERSION}")

@main.command()
def doctor():
    """Check ARF installation and dependencies"""
    click.echo("Checking ARF installation...")
    
    # Check FAISS (but only when needed)
    try:
        import faiss
        click.echo("✓ FAISS installed")
    except ImportError:
        click.echo("✗ FAISS not installed", err=True)
        sys.exit(1)
    
    # Check other deps
    deps = [
        ("SentenceTransformers", "sentence_transformers"),
        ("Gradio", "gradio"),
        ("Pandas", "pandas"),
        ("NumPy", "numpy"),
        ("Pydantic", "pydantic"),
        ("Requests", "requests"),
        ("CircuitBreaker", "circuitbreaker"),
        ("atomicwrites", "atomicwrites"),
        ("python-dotenv", "dotenv"),
        ("Click", "click"),
    ]
    
    all_ok = True
    for name, module in deps:
        try:
            __import__(module)
            click.echo(f"  ✓ {name}")
        except ImportError:
            click.echo(f"  ✗ {name}")
            all_ok = False
    
    if all_ok:
        click.echo("\n✅ All dependencies OK!")
    else:
        click.echo("\n❌ Some dependencies missing")
        sys.exit(1)

@main.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=7860, type=int, help='Port to bind to')
@click.option('--share/--no-share', default=False, help='Create public Gradio share link')
def serve(host, port, share):
    """Start the ARF Gradio UI server (loads FAISS)"""
    click.echo(f"🚀 Starting ARF v{VERSION} on {host}:{port}...")
    
    # NOW import agentic_reliability_framework as arf and load FAISS
    import agentic_reliability_framework as arf
    demo = arf.create_enhanced_ui()
    demo.launch(server_name=host, server_port=port, share=share)

if __name__ == "__main__":
    main()
