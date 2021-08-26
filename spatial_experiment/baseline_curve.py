import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.export_model as export_model
import jiant.proj.main.runscript as main_runscript
import jiant.utils.python.io as py_io
import jiant.utils.display as display
import shutil
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


def execute_training_loop():
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


train_path = 'spatial_experiment/tasks/data/spatial/train.jsonl'
val_path = 'spatial_experiment/tasks/data/spatial/val.jsonl'
test_path = 'spatial_experiment/tasks/data/spatial/test.jsonl'
tr_vl_te = [train_path, val_path, test_path]

d = '../spatial-commonsense/data/baseline'
baseline_path = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]

for dir in baseline_path:
    dir_name = dir.split('/')[-1]
    with open(dir_name + '.csv', 'w') as f:
        f.write('name,acc,f1,recall,precision\n')
    print('wrote:', dir_name + '.csv')
    all_s_paths = [os.path.join(dir, o) for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    split_paths = [x for x in all_s_paths if 'baseline' in x]
    for path in tr_vl_te:
        if os.path.isfile(path):
            os.remove(path)
    shutil.copyfile(os.path.join(dir, 'train.jsonl'), train_path)
    shutil.copyfile(os.path.join(dir, 'val.jsonl'), val_path)
    shutil.copyfile(os.path.join(dir, 'test.jsonl'), test_path)
    execute_training_loop()
    with open('output.csv', 'r') as f:
        output = f.read()
    max_test_path = os.path.join(dir, 'test.jsonl')
    count = sum(1 for _ in open(max_test_path, encoding='utf8'))
    print(f'Number of rows in {max_test_path}: {count}')
    with open(dir_name + '.csv', 'a') as f:
        f.write(f'{str(count)},{output}')
    print(f'wrote: {str(count)},{output.strip()}')
    if os.path.isfile('output.csv'):
        os.remove('output.csv')
    for split_dir in split_paths:
        split_dir_name = split_dir.split('/')[-1]
        if os.path.isfile(train_path):
            os.remove(train_path)
        shutil.copyfile(os.path.join(split_dir, 'train.jsonl'), train_path)
        execute_training_loop()
        with open('output.csv', 'r') as f:
            output = f.read()
        with open(dir_name + '.csv', 'a') as f:
            f.write(f'{split_dir_name},{output}')
        print(f'wrote: {split_dir_name},{output.strip()}')
        if os.path.isfile('output.csv'):
            os.remove('output.csv')
