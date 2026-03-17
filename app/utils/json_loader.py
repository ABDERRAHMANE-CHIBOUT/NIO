import json

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def json_to_text(data):
    return json.dumps(data, indent=2)