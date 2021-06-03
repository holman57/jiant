
import sys
sys.path.insert(0, "../jiant")

import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.export_model as export_model
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.runscript as main_runscript
import jiant.shared.caching as caching
import jiant.utils.python.io as py_io
import jiant.utils.display as display
import os

run_args = main_runscript.RunConfiguration(
    jiant_task_container_config_path="./run_configs/spatial_run_config.json",
    output_dir="./runs/spatial",
    hf_pretrained_model_name_or_path="bert-base-uncased",
    model_path="./models/bert/model/model.p",
    model_config_path="./models/bert/model/config.json",
    learning_rate=1e-5,
    eval_every_steps=500,
    do_train=False,
    do_val=True,
    do_save=True,
    force_overwrite=True,
)

main_runscript.run_loop(run_args)


