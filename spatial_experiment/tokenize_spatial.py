import jiant.proj.main.tokenize_and_cache as tokenize_and_cache

tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
    task_config_path=f"./spatial_config.json",
    hf_pretrained_model_name_or_path="bert-base-uncased",
    output_dir=f"./cache/spatial",
    phases=["train", "val"],
))
