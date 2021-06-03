import jiant.proj.main.runscript as main_runscript

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
