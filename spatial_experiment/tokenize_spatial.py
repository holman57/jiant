
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

tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
    task_config_path=f"./tasks/configs/spatial_config.json",
    hf_pretrained_model_name_or_path="bert-base-uncased",
    output_dir=f"./cache/spatial",
    phases=["train", "val"],
))



