import re

# Read the file
with open('app.py', 'r') as f:
    content = f.read()

# Fix all gr.Dataframe returns to gr.update
patterns_to_fix = [
    # Fix rate limit error return (line ~2084)
    (r'return \(\s*rate_msg,\s*{},\s*{},\s*gr\.Dataframe\(value=\[\]\),',
     'return (rate_msg, {}, {}, gr.update(value=[]),'),
    
    # Fix type conversion error (line ~2105)
    (r'return \(\s*error_msg,\s*{},\s*{},\s*gr\.Dataframe\(value=\[\]\),',
     'return (error_msg, {}, {}, gr.update(value=[]),'),
    
    # Fix validation error (line ~2122)
    (r'return \(\s*error_msg,\s*{},\s*{},\s*gr\.Dataframe\(value=\[\]\),',
     'return (error_msg, {}, {}, gr.update(value=[]),'),
    
    # Fix processing error (line ~2140)
    (r'return \(\s*f"❌ \{result\[\'error\'\]}",\s*{},\s*{},\s*gr\.Dataframe\(value=\[\]\),',
     'return (f"❌ {result[\'error\']}", {}, {}, gr.update(value=[]),'),
    
    # Fix main success case (lines 2202-2208)
    (r'gr\.Dataframe\(\s*headers=\["Timestamp", "Component", "Latency", "Error Rate", "Throughput", "Severity", "Analysis"\],\s*value=table_data,\s*wrap=True\s*\)',
     'gr.update(value=table_data)'),
    
    # Fix exception handler (line after 2220)
    (r'error_msg,\s*{},\s*{},\s*gr\.Dataframe\(value=\[\]\),',
     'error_msg, {}, {}, gr.update(value=[]),')
]

for pattern, replacement in patterns_to_fix:
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

# Write the fixed file
with open('app.py.fixed', 'w') as f:
    f.write(content)

print("Fixed file saved as app.py.fixed")
print("To apply: cp app.py.fixed app.py")
