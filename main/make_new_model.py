import os
import shutil
import sys
import subprocess
import re
import random
import importlib.util
from improve_code import improve_file_chunks, process_file, process_file_local


def get_next_model_dir(base_dir: str = "models", prefix: str = "model") -> str:
    existing = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and re.match(f"{prefix}\\d+", d)
    ]
    numbers = [int(re.findall(r"\d+", d)[0]) for d in existing]
    next_number = max(numbers, default=0) + 1
    return os.path.join(base_dir, f"{prefix}{next_number}")


def duplicate_model_dir(src_dir: str, base_dir: str = "models", prefix: str = "model", overwrite: bool = False):
    dst_dir = get_next_model_dir(base_dir, prefix)
    if os.path.exists(dst_dir):
        if overwrite:
            shutil.rmtree(dst_dir)
        else:
            raise FileExistsError(f"Destination already exists: {dst_dir}")
    shutil.copytree(src_dir, dst_dir)
    return dst_dir


def improve_all_files(model_dir: str, backend: str = "openai"):
    excluded = {"sample.py", "bench.py"}

    for filename in os.listdir(model_dir):
        if filename.endswith(".py") and filename not in excluded:
            file_path = os.path.join(model_dir, filename)

            try:
                process_file(file_path, model_dir, backend)
                # or if using improve_model_file:
                # improve_file_chunks(model_dir, filename, improved_filename, backend=backend)
            except Exception as e:
                print(f"Failed to improve {filename}: {e}")


def improve_all_files_local(model_dir: str, modifier_model_dir: str):
    excluded = {"sample.py", "bench.py"}

    for filename in os.listdir(model_dir):
        if filename.endswith(".py") and filename not in excluded:
            file_path = os.path.join(model_dir, filename)
            improved_filename = os.path.join(model_dir, filename)

            try:
                process_file_local(file_path, model_dir, modifier_model_dir)
                # or if using improve_model_file:
                # improve_file_chunks(model_dir, filename, improved_filename, backend=backend)
            except Exception as e:
                print(f"Failed to improve {filename}: {e}")

def run_benchmark(model_dir):
    bench_path = os.path.join(model_dir, "bench.py")
    if not os.path.exists(bench_path):
        print(f"[!] No bench.py found in {model_dir}, skipping benchmark.")
        return None, None

    sys.path.insert(0, model_dir)
    cwd = os.getcwd()

    try:
        os.chdir(model_dir)
        spec = importlib.util.spec_from_file_location("bench", bench_path)
        bench_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench_module)

        time_ms, mfu = bench_module.run_benchmark()
        return time_ms, mfu
    except Exception as e:
        print(f"[x] Error running benchmark: {e}")
        return None, None
    finally:
        os.chdir(cwd)
        if model_dir in sys.path:
            sys.path.remove(model_dir)



def improve_loop(src_model_name: str = "model1", backend: str = "openai"):

    project_root = os.path.dirname(os.path.abspath(__file__))
    src_model_dir = os.path.join(project_root, "models", src_model_name)

    # Step 1: Duplicate model directory
    try:
        dst_model_dir = duplicate_model_dir(src_model_dir, base_dir=os.path.join(project_root, "models"))
    except Exception as e:
        print(f"[x] Failed to duplicate model: {e}")
        return

    # Step 2: Improve all Python files
    try:
        improve_all_files(dst_model_dir, backend=backend)
    except Exception as e:
        print(f"[x] Error improving Python files in {dst_model_dir}: {e}")
        return

    # Step 3: Run train.py in the duplicated directory
    train_script = os.path.join(dst_model_dir, "train.py")
    command = [sys.executable, train_script]

    try:
        subprocess.run(command, cwd=dst_model_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[x] Training script exited with error code: {e.returncode}")
    except FileNotFoundError:
        print(f"[x] Could not find train.py in {dst_model_dir}.")
    except Exception as e:
        print(f"[x] Unexpected error running train.py: {e}")

    # Step 4: Run benchmark
    time_ms, mfu = run_benchmark(dst_model_dir)
    if time_ms is not None:
        print(f"{dst_model_dir} Benchmark results — time per iter: {time_ms:.4f} ms, MFU: {mfu:.2f}%")

    return time_ms, mfu


def improve_loop_local(src_model_name: str, modifier_model_name: str):
    project_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_root, "models")
    src_model_dir = os.path.join(models_dir, src_model_name)
    modifier_model_dir = os.path.join(models_dir, modifier_model_name)

    # Step 1: Duplicate the source model
    try:
        dst_model_dir = duplicate_model_dir(src_model_dir, base_dir=models_dir)
    except Exception as e:
        print(f"[x] Failed to duplicate source model: {e}")
        return None, None

    # Step 2: Apply local modifications based on modifier_model_dir
    try:
        improve_all_files_local(dst_model_dir, modifier_model_dir)
    except Exception as e:
        print(f"[x] Error applying local modifications: {e}")
        return None, None

    # Step 3: Run train.py
    train_script = os.path.join(dst_model_dir, "train.py")
    command = [sys.executable, train_script]

    try:
        subprocess.run(command, cwd=dst_model_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[x] Training script exited with error code: {e.returncode}")
    except FileNotFoundError:
        print(f"[x] Could not find train.py in {dst_model_dir}.")
    except Exception as e:
        print(f"[x] Unexpected error running train.py: {e}")

    # Step 4: Run benchmark directly
    time_ms, mfu = run_benchmark(dst_model_dir)
    if time_ms is not None:
        print(f"{dst_model_dir} Benchmark results — time per iter: {time_ms:.4f} ms | MFU: {mfu:.2f}%")

    return time_ms, mfu


def collect_performance(model_name):
    project_root = os.path.dirname(os.path.abspath(__file__))
    dst_model_dir = os.path.join(project_root, "models", model_name)
    time_ms, mfu = run_benchmark(dst_model_dir)
    if time_ms is not None:
        print(f"Benchmark results — time per iter: {time_ms:.4f} ms | MFU: {mfu:.2f}%")

    return time_ms, mfu


def list_models(base_dir="models", prefix="model"):
    models = []
    for d in os.listdir(base_dir):
        path = os.path.join(base_dir, d)
        if os.path.isdir(path):
            match = re.match(f"{prefix}(\\d+)$", d)
            if match:
                models.append((int(match[1]), d))
    return sorted(models, key=lambda x: x[0])


def delete_model(model_number: int, base_dir="models", prefix="model"):
    models = list_models(base_dir, prefix)
    numbers = [num for num, _ in models]

    if model_number not in numbers:
        raise ValueError(f"{prefix}{model_number} does not exist.")

    target_name = f"{prefix}{model_number}"
    target_path = os.path.join(base_dir, target_name)

    last_number, last_name = models[-1]
    last_path = os.path.join(base_dir, last_name)

    shutil.rmtree(target_path)

    if model_number != last_number:
        new_path = os.path.join(base_dir, target_name)
        shutil.move(last_path, new_path)

    return target_path


def ga_selection(performance, num_parents=4, num_to_cull=6):
    # Filter out invalid MFUs
    valid = [(num, mfu) for (num, _, mfu) in performance if mfu is not None]
    invalid = [(num, mfu) for (num, _, mfu) in performance if mfu is None]

    if not valid:
        print("No valid MFU scores; random selection.")
        all_nums = [num for (num, _, _) in performance]
        return random.sample(all_nums, num_parents), all_nums[-num_to_cull:]

    # Normalize MFU scores
    total_mfu = sum(mfu for _, mfu in valid)
    if total_mfu == 0:
        probs = [1 / len(valid)] * len(valid)
    else:
        probs = [mfu / total_mfu for _, mfu in valid]

    model_nums = [num for num, _ in valid]

    # Select parents
    selected_parents = random.choices(model_nums, weights=probs, k=num_parents)
    selected_parents = list(set(selected_parents))  # avoid duplicates if possible

    # Find models to delete (lowest MFU or None)
    sorted_perf = sorted(performance, key=lambda x: (x[2] if x[2] is not None else -1))
    to_delete = [num for (num, _, _) in sorted_perf[:num_to_cull]]

    return selected_parents, to_delete
