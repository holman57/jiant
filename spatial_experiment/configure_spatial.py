import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.export_model as export_model
import jiant.utils.python.io as py_io
import jiant.utils.display as display
import os

jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
    task_config_base_path="./spatial_experiment",
    task_cache_base_path="./spatial_experiment/cache",
    train_task_name_list=["spatial"],
    val_task_name_list=["spatial"],
    train_batch_size=8,
    eval_batch_size=16,
    epochs=3,
    num_gpus=1,
).create_config()

os.makedirs("./spatial_experiment/run_configs/", exist_ok=True)
py_io.write_json(jiant_run_config, "./spatial_experiment/run_configs/spatial_run_config.json")
display.show_json(jiant_run_config)

export_model.export_model(
    hf_pretrained_model_name_or_path="bert-base-uncased",
    output_base_path="./spatial_experiment/models/bert",
)
