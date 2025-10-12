from make_new_model import list_models
from concurrent.futures import ProcessPoolExecutor, as_completed
from requests_and_memory import *
import random
import os

ITERS = 1

# Determines how many models to keep each generation.
KEPT_MODELS = ["model1", "model2", "model3", "model4"]

# local or openai, note that some new features have not yet been implemented for local.
BACKEND = "openai"

def parallel_improve(src_model_name, backend):
    try:
        return src_model_name, *improve_loop(src_model_name=src_model_name, backend=backend)
    except Exception as e:
        print(f"Error improving {src_model_name}: {e}")
        return src_model_name, None, None, None


def main(existing_models, total_models=10, backend="openai", iters=2, max_workers=2):
    project_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_root, "models")

    for i in range(iters):
        print(f"\nIteration {i + 1}/{iters}")
        performance = []

        current_models = [d for d in os.listdir(models_dir)
                          if os.path.isdir(os.path.join(models_dir, d)) and d.startswith("model")]

        # Step 1: Collect performance
        for idx, model_name in enumerate(current_models):
            time_ms, mfu = collect_performance(model_name)
            performance.append((int(model_name.replace("model", "")), time_ms, mfu))

        # Step 2: Duplicate until total_models
        if len(current_models) < total_models:
            src_models = [random.choice(existing_models) for _ in range(total_models - len(current_models))]

            print(f"Running in multithread.")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                completed = [
                    executor.submit(parallel_improve, src_model, backend)
                    for src_model in src_models
                ]
                for future in as_completed(completed):
                    try:
                        src, time_ms, mfu, dst = future.result()
                        if not dst:
                            print(f"Worker for {src} returned no destination.")
                            continue

                        model_name = os.path.basename(dst)
                        new_idx = int(model_name.replace("model", ""))
                        performance.append((new_idx, time_ms, mfu))

                    except Exception as e:
                        print(f"Worker failed: {e}")

        # Step 3: Genetic Algorithm
        parents, to_delete = ga_selection(performance, num_parents=4, num_to_cull=6)

        for num in sorted(to_delete, reverse=True):
            try:
                delete_model(num, base_dir=models_dir)
            except Exception as e:
                print(f"Failed to delete model{num}: {e}")

        print(f"Kept: {list_models(models_dir)}")


if __name__ == "__main__":
    main(KEPT_MODELS, iters=ITERS, backend=BACKEND)

    project_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_root, "models")
    collect_model_requests(models_dir)