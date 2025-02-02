import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.export_model as export_model
import jiant.proj.main.runscript as main_runscript
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

train_path = 'spatial_experiment/tasks/data/spatial/train.jsonl'
val_path = 'spatial_experiment/tasks/data/spatial/val.jsonl'
test_path = 'spatial_experiment/tasks/data/spatial/test.jsonl'
tr_vl_te = [train_path, val_path, test_path]

d = '../spatial-commonsense/data/baseline'

with open('error.csv', 'w') as f:
    f.write('sentence,label,prediction\n')
print('wrote: error.csv')

tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
    task_config_path=f"./spatial_experiment/spatial_config.json",
    hf_pretrained_model_name_or_path="bert-base-uncased",
    output_dir=f"./spatial_experiment/cache/spatial",
    phases=["train", "val"],
))

run_args = main_runscript.RunConfiguration(
    jiant_task_container_config_path="./spatial_experiment/run_configs/spatial_run_config.json",
    output_dir="./spatial_experiment/runs/spatial",
    hf_pretrained_model_name_or_path="bert-base-uncased",
    model_path="./spatial_experiment/models/bert/model/model.p",
    model_config_path="./spatial_experiment/models/bert/model/config.json",
    learning_rate=1e-5,
    eval_every_steps=500,
    do_train=True,
    do_val=True,
    do_save=True,
    force_overwrite=True,
)

main_runscript.run_loop(run_args)

with open(val_path, 'r') as f:
    test_file = f.readlines()
test_file.pop()
for test in test_file:
    with open(val_path, 'w') as f:
        f.write(f'guid,label,sentence\n{test}')
    tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
        task_config_path=f"./spatial_experiment/spatial_config.json",
        hf_pretrained_model_name_or_path="bert-base-uncased",
        output_dir=f"./spatial_experiment/cache/spatial",
        phases=["train", "val"],
    ))
    run_args = main_runscript.RunConfiguration(
        jiant_task_container_config_path="./spatial_experiment/run_configs/spatial_run_config.json",
        output_dir="./spatial_experiment/runs/spatial",
        hf_pretrained_model_name_or_path="bert-base-uncased",
        model_path="./spatial_experiment/models/bert/model/model.p",
        model_config_path="./spatial_experiment/models/bert/model/config.json",
        learning_rate=1e-5,
        eval_every_steps=500,
        do_train=False,
        do_val=False,
        do_save=False,
        force_overwrite=False,
    )
    main_runscript.run_loop(run_args)
    with open('output.csv', 'r') as f:
        prediction = f.read()
    with open('error.csv', 'a') as f:
        sentence = test.split(",")[2].replace("\n", "")
        f.write(f'{sentence},{test.split(",")[1]},{prediction.split(",")[1]}\n')
