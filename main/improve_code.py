import os
import sys
import re
import time
import traceback
import subprocess
import tiktoken
import random


MAX_INPUT_CHARS = 30000


def process_file(model_file, model_dir, backend, MAX_INPUT_TOKENS=8000, MAX_OUTPUT_TOKENS=8000):
    # Step 1: Load and sanitize
    orig_code = read_source(model_file)
    orig_code = sanitize_text(orig_code)

    # Step 2: Tokenization
    try:
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    input_tokens = enc.encode(orig_code)

    if len(input_tokens) > MAX_INPUT_TOKENS:
        print(f"[!] ERROR: Input file '{model_file}' is too large ({len(input_tokens)} tokens, limit {MAX_INPUT_TOKENS}). Skipping.")
        return None, {
            "error": "input_too_large",
            "input_tokens": len(input_tokens),
            "source_file": model_file,
            "backend": backend,
        }

    # Step 3: Prompting
    prompt = prompt_template(orig_code, False)

    timestamp = int(time.time())
    meta = {
        "timestamp": timestamp,
        "backend": backend,
        "source_file": model_file,
        "input_tokens": len(input_tokens),
    }

    # Step 5: Model inference
    try:
        if backend == "openai":
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            generated = openai_completion(
                prompt,
                model_name=model_name,
                max_tokens=MAX_OUTPUT_TOKENS
            )
        else:
            generated = use_local_model_completion(
                prompt,
                sample_script=os.path.join(model_dir, model_file)
            )
    except Exception as e:
        print(f"[!] Generation failed for {model_file}: {e}")
        traceback.print_exc()
        meta["error"] = str(e)
        return None, meta

    # Step 6: Extract improved section
    marker = "# Improved training code:"
    if marker in generated:
        improved_code = generated.split(marker, 1)[1].strip()
    else:
        improved_code = generated.strip()

    output_tokens = enc.encode(improved_code)
    out_len = len(output_tokens)

    if out_len > MAX_OUTPUT_TOKENS:
        print(f"[!] ERROR: Output for '{model_file}' exceeds {MAX_OUTPUT_TOKENS} tokens ({out_len}). Skipping save.")
        meta["error"] = "output_too_large"
        meta["output_tokens"] = out_len
        return None, meta

    meta["output_tokens"] = out_len
    return improved_code, meta


def process_file_local(model_file, model_dir, backend_dir, MAX_INPUT_TOKENS=8000, MAX_OUTPUT_TOKENS=8000):
    # Step 1: Load and sanitize
    orig_code = read_source(model_file)
    orig_code = sanitize_text(orig_code)

    # Step 2: Tokenization
    try:
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    input_tokens = enc.encode(orig_code)
    token_len = len(input_tokens)

    if token_len > MAX_INPUT_TOKENS:
        print(f"[!] ERROR: Input file '{model_file}' is too large ({token_len} tokens, limit {MAX_INPUT_TOKENS}). Skipping.")
        return None, {
            "error": "input_too_large",
            "input_tokens": token_len,
            "source_file": model_file,
            "backend": backend_dir,
        }

    # Step 3: Prompting
    prompt = prompt_template(orig_code, False)

    timestamp = int(time.time())
    meta = {
        "timestamp": timestamp,
        "backend": backend_dir,
        "source_file": model_file,
        "input_tokens": token_len,
    }

    # Step 5: Model inference
    try:
        generated = use_local_model_completion(
            prompt,
            sample_script=os.path.join(model_dir, model_file, backend_dir)
        )
    except Exception as e:
        print(f"[!] Generation failed for {model_file}: {e}")
        traceback.print_exc()
        meta["error"] = str(e)
        return None, meta

    # Step 6: Extract improved section
    marker = "# Improved training code:"
    if marker in generated:
        improved_code = generated.split(marker, 1)[1].strip()
    else:
        improved_code = generated.strip()

    output_tokens = enc.encode(improved_code)
    out_len = len(output_tokens)

    if out_len > MAX_OUTPUT_TOKENS:
        print(f"[!] ERROR: Output for '{model_file}' exceeds {MAX_OUTPUT_TOKENS} tokens ({out_len}). Skipping save.")
        meta["error"] = "output_too_large"
        meta["output_tokens"] = out_len
        return None, meta

    meta["output_tokens"] = out_len
    return improved_code, meta


def read_source(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def split_into_chunks(code: str):
    # Match top-level `class` or `def`
    pattern = re.compile(r'^(class\s+\w+\s*(\(.*?\))?\s*:|def\s+\w+\s*\(.*?\)\s*:)', re.M)
    matches = list(pattern.finditer(code))
    chunks = []

    if not matches:
        return [("module_level", code.strip())]

    first_start = matches[0].start()
    preamble = code[:first_start].strip()
    if preamble:
        chunks.append(("module_level", preamble))

    # Extract each class or function chunk
    for i, m in enumerate(matches):
        header = m.group(1).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(code)
        body = code[start:end].rstrip()
        chunks.append((header, body))

    return chunks


def clean_chunk(chunk: str) -> str:
    lines = [line.rstrip() for line in chunk.strip().splitlines()]

    while lines and (lines[0].startswith("'''") or lines[0].startswith('"""')
                     or lines[0].startswith("```") or lines[0].startswith("#")):
        lines = lines[1:]

    while lines and (lines[-1].startswith("'''") or lines[-1].startswith('"""')
                     or lines[-1].startswith("```") or lines[-1].startswith("#")):
        lines = lines[:-1]

    return "\n".join(lines).strip()


def sanitize_text(s: str) -> str:
    return s.replace("\r", "").replace("\x00", "")


def truncate_text(s: str, max_chars: int) -> (str, bool):
    if len(s) <= max_chars:
        return s, False
    head = s[: max_chars // 2]
    tail = s[- (max_chars // 2) : ]
    truncated = head + "\n\n# ...TRUNCATED... \n\n" + tail
    return truncated, True


def prompt_template(original_code: str, truncated_flag: bool):
    trunc_note = ""
    if truncated_flag:
        trunc_note = (
            "\n# NOTE: The code was truncated to fit token limits. "
            "Improve only the visible part.\n"
        )

    return f"""
You are a code optimizer specializing in deep learning and PyTorch.
Make small, local improvements to the following Python code fragment.

Focus strictly on:
- GPU/memory efficiency
- numerical stability
- concise, clean style

Do NOT:
- add print statements or docstrings
- include markdown, quotes, or explanations
- change the overall structure or functionality
- output anything except raw, valid Python code

Output only valid Python source code.
No text or comments outside the code itself.

# BEGIN ORIGINAL
{original_code}
# END ORIGINAL

# BEGIN IMPROVED
{trunc_note}
"""

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def openai_completion(prompt: str, model_name: str = "gpt-4o-mini", temperature: float = 0.2, max_tokens: int = 4000):
    if OpenAI is None:
        raise RuntimeError("OpenAI library not installed or failed to import.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": "You are an expert deep learning engineer."},
        {"role": "user", "content": prompt},
    ]
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    try:
        content = resp.choices[0].message.content
    except Exception:
        content = getattr(resp, "choices", [None])[0]
        if content and isinstance(content, dict):
            content = content.get("message", {}).get("content", "")
        else:
            content = str(resp)
    return content


def use_local_model_completion(
        prompt: str,
        sample_script: str = "models/model1/train.py",
        backend_dir: str = "models/model1",
        start_token: str = "",
        num_samples: int = 1,
        max_new_tokens: int = 4096
):

    sample_path = os.path.join(backend_dir, "sample.py")
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"sample.py not found in {backend_dir}")

    cmd = [
        sys.executable, sample_path,
        "--max_new_tokens", str(max_new_tokens),
        "--num_samples", str(num_samples)
    ]

    try:
        completed = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            cwd=backend_dir,
            check=True
        )
        generated = completed.stdout.strip()
        return generated

    except subprocess.CalledProcessError as e:
        print(f"[x] Error running local model ({backend_dir}):\n{e.stderr}")
        return None


def save_output(code: str, out_code_file: str):
    lines = [line.rstrip() for line in code.strip().splitlines()]
    while lines and (lines[0].startswith("'''") or lines[0].startswith('"""') or lines[0].startswith("```") or lines[0].startswith("#")):
        lines = lines[1:]

    while lines and (lines[-1].startswith("'''") or lines[-1].startswith('"""') or lines[-1].startswith("```") or lines[-1].startswith("#")):
        lines = lines[:-1]

    code = "\n".join(lines).strip()
    with open(out_code_file, "w", encoding="utf-8") as f:
        f.write(code)


def improve_file_chunks(model_dir: str, model_filename: str, improved_filename: str, backend: str = "openai", modify_prob: float = 0.4):
    """
    Improves a Python model/training file by chunking it (by class/function)
    and generating improved code for each chunk.

    Args:
        model_dir: Path to the directory containing the model file.
        model_filename: Name of the file to process.
        backend: "openai" or "local". If None, auto-selects based on environment.

    Returns:
        Tuple of (final_code: str or None, meta: dict)
    """
    model_file = os.path.join(model_dir, model_filename)
    if not os.path.exists(model_file):
        error_msg = f"Error: {model_filename} not found in {model_dir}."
        print(error_msg)
        return None, {"error": error_msg}

    # Determine backend
    if backend is None:
        backend = os.getenv("BACKEND", "").lower()
        if backend == "":
            backend = "openai" if os.getenv("OPENAI_API_KEY") else "local"
    print(f"Selected backend: {backend}")

    # Read and chunk code
    orig_code = read_source(model_file)
    chunks = split_into_chunks(orig_code)

    improved_chunks = []
    prev_function_names = []

    # Process chunks sequentially
    for idx, (header, chunk) in enumerate(chunks, start=1):
        if random.random() > modify_prob or header == "module_level":
            improved_chunks.append(chunk)
            continue

        class_name_match = re.search(r'class\s+(\w+)', header)
        chunk = clean_chunk(chunk)
        chunk_trunc, was_truncated = truncate_text(chunk, MAX_INPUT_CHARS)

        current_name = class_name_match.group(1) if class_name_match else header

        context_summary = ""
        if idx > 1:
            context_summary = "\n# Previously improved sections:\n" + "\n".join(prev_function_names[-3:]) + "\n"

        # Module-level imports/constants
        module_chunk = next((body for h, body in chunks if h == "module_level"), None)
        import_summary = f"\n# Module-level imports and constants:\n{module_chunk}\n" if module_chunk else ""

        prompt = context_summary + import_summary + prompt_template(chunk_trunc, was_truncated)

        try:
            if backend == "openai":
                generated = openai_completion(prompt)
            else:
                generated = use_local_model_completion(prompt, sample_script=os.path.join(model_dir, "sample.py"))
        except Exception as e:
            print(f"[!] Chunk {idx} failed: {e}")
            continue

        marker = "# Improved training code:"
        improved_code = generated.split(marker, 1)[1].strip() if marker in generated else generated.strip()
        improved_code = clean_chunk(improved_code)
        improved_chunks.append(improved_code)

        name_match = re.search(r'class\s+(\w+)', header)
        if name_match:
            prev_function_names.append(name_match.group(1))

    # Merge improved chunks
    merged_code = "\n\n".join(improved_chunks)
    save_output(merged_code, improved_filename)

    final_code, meta = process_file(model_file, model_dir, backend)
    if final_code is None:
        print(f"Skipped file due to error: {meta.get('error', 'unknown error')}")
        return None, meta

    save_output(final_code, improved_filename)
    return final_code, meta


if __name__ == "__main__":
    improved_filename = "models/model1/model2.py"
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(project_root, "models", "model1")
    improve_file_chunks(model_dir, "model.py", improved_filename, backend="openai")
