import jiant.proj.main.tokenize_and_cache as tokenize_and_cache

tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
    task_config_path=f"./spatial_experiment/spatial_config.json",
    hf_pretrained_model_name_or_path="bert-base-uncased",
    output_dir=f"./spatial_experiment/cache/spatial",
    phases=["train", "val"],
))
