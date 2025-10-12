import os
from make_new_model import *
from openai import OpenAI
import json
from datetime import datetime
from pathlib import Path

def get_dir_tree(path, prefix=""):
    entries = os.listdir(path)
    entries.sort()
    tree_str = ""
    for i, entry in enumerate(entries):
        entry_path = os.path.join(path, entry)
        connector = "└── " if i == len(entries) - 1 else "├── "
        tree_str += f"{prefix}{connector}{entry}\n"
        if os.path.isdir(entry_path):
            extension = "    " if i == len(entries) - 1 else "│   "
            tree_str += get_dir_tree(entry_path, prefix + extension)
    return tree_str


def query_model_for_requests(model_name, model_path, backend="openai"):
    """
    Ask the model for improvement suggestions (simulated via OpenAI API for now).
    """
    dir_tree = get_dir_tree(model_path)
    client = OpenAI()

    prompt = f"""
You are acting as an autonomous AI model named {model_name}.
Your project directory structure is:

{dir_tree}

Based on this, propose improvements in the following categories:
1. Dataset (e.g., need more or cleaner data)
2. Scripts or functions (e.g., new analysis, efficiency changes)
3. File structure (e.g., reorganize scripts/models)
4. Libraries (e.g., additional installs or version changes)

Be concise and structured. Only choose one of the 4 options. If requesting a script, output the full code for the script.
"""

    if backend == "openai":
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    else:
        return f"{model_name}: Could not return prompt."


def collect_model_requests(models_dir, backend="openai", output_file="model_requests.txt"):

    remaining = list_models(models_dir)
    results = []

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("")

    with open(output_file, "a", encoding="utf-8") as f:
        for num, model_name in remaining:
            model_path = os.path.join(models_dir, model_name)

            try:
                feedback = query_model_for_requests(model_name, model_path, backend)
                entry = f"\n=== {model_name} ===\n{feedback}\n"
                f.write(entry)
                f.flush()
                results.append((model_name, feedback))
            except Exception as e:
                print(f"Failed to query {model_name}: {e}")
    print(f"Requests saved in {output_file}")
    return results


def save_to_memory(model_file, summary, meta, memory_dir="memory", base_dir="models"):
    model_name = Path(model_file).stem if Path(model_file).suffix else Path(model_file).name
    model_dir = Path(base_dir) / model_name / memory_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"memory_{timestamp}.jsonl"
    file_path = model_dir / filename

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "file": model_file,
        "summary": summary,
        **meta
    }

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return str(file_path)
