import re

with open('agentic_reliability_framework/app.py', 'r') as f:
    content = f.read()

# Fix line 1656: thread_safe_index should check if get_faiss_index() returns something
content = re.sub(
    r'"similar_incidents_count": get_faiss_index\(\)\.get_count\(\) if thread_safe_index and is_anomaly else 0,',
    '"similar_incidents_count": get_faiss_index().get_count() if is_anomaly else 0,',
    content
)

# Fix line 2276: same issue
content = re.sub(
    r'logger\.info\(f"Vector index size: \{get_faiss_index\(\)\.get_count\(\) if thread_safe_index else 0\}"\)',
    'logger.info(f"Vector index size: {get_faiss_index().get_count()}")',
    content
)

# Also need to fix line 1626 which checks if get_faiss_index() is not None
# get_faiss_index() returns the index object, not None when not initialized
# Let me check what get_faiss_index() actually returns
content = re.sub(
    r'if get_faiss_index\(\) is not None and model is not None and is_anomaly:',
    'if model is not None and is_anomaly:',
    content
)

with open('agentic_reliability_framework/app.py', 'w') as f:
    f.write(content)

print("Fixed thread_safe_index issues in app.py")
