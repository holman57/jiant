import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.export_model as export_model
import jiant.utils.python.io as py_io
import jiant.utils.display as display
import os

jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
    task_config_base_path="./tasks/configs",
    task_cache_base_path="./cache",
    train_task_name_list=["spatial"],
    val_task_name_list=["spatial"],
    train_batch_size=8,
    eval_batch_size=16,
    epochs=3,
    num_gpus=1,
).create_config()

os.makedirs("./run_configs/", exist_ok=True)
py_io.write_json(jiant_run_config, "./run_configs/spatial_run_config.json")
display.show_json(jiant_run_config)
