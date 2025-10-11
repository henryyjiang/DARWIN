from make_new_model import *


def main(existing_models, total_models=10, backend="openai", iterations=2):
    project_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_root, "models")

    for i in range(iterations):
        print(f"\nIteration {i + 1}/{iterations}")
        performance = []

        # Refresh model list
        current_models = [d for d in os.listdir(models_dir)
                          if os.path.isdir(os.path.join(models_dir, d)) and d.startswith("model")]

        # Step 1: Collect performance
        for idx, model_name in enumerate(current_models):
            time_ms, mfu = collect_performance(model_name, backend=backend)
            performance.append((int(model_name.replace("model", "")), time_ms, mfu))

        # Step 2: Expand until total_models
        while len(current_models) < total_models:
            src_model_name = random.choice(existing_models)
            time_ms, mfu = improve_loop(src_model_name=src_model_name, backend=backend)
            new_idx = len(current_models) + 1
            performance.append((new_idx, time_ms, mfu))

            current_models = [d for d in os.listdir(models_dir)
                              if os.path.isdir(os.path.join(models_dir, d)) and d.startswith("model")]

        # Step 3: Genetic Algorithm
        parents, to_delete = ga_selection(performance, num_parents=4, num_to_cull=6)

        for num in sorted(to_delete, reverse=True):
            try:
                delete_model(num, base_dir=models_dir)
            except Exception as e:
                print(f"Failed to delete model{num}: {e}")

        remaining = list_models(models_dir)
        print(f"Kept: {remaining}")

if __name__ == "__main__":
    main(["model1", "model2", "model3", "model4"])