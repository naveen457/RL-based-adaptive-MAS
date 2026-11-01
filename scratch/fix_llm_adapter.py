import os

def fix():
    path = os.path.join('app', 'architecture', 'llm_adapter.py')
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Replace lines 263-285 (1-based)
    # Let's inspect the lines exactly
    for i, l in enumerate(lines):
        if 'for attempt in range(3):' in l:
            idx = i
            break
            
    correct_block = """        for attempt in range(3):
            try:
                response = self.client.invoke(prompt)
                break
            except Exception as e:
                err_str = str(e).lower()
                if any(w in err_str for w in ["getaddrinfo", "connecterror", "connection error", "connectionpool", "connection refused", "winerror 10013"]):
                    break
                if ("rate_limit" in err_str or "429" in err_str or "tokens per minute" in err_str) and attempt < 2:
                    print("  [Rate Limit] Replenishing tokens, waiting 5s...")
                    time.sleep(5)
                    continue
                if attempt == 2:
                    break
"""
    # find where "if response is None:" is
    for i in range(idx, len(lines)):
        if 'if response is None:' in lines[i]:
            end_idx = i
            break
            
    new_lines = lines[:idx] + [correct_block] + lines[end_idx:]
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(new_lines)
    print("Fixed indentation in llm_adapter.py successfully!")

if __name__ == '__main__':
    fix()

