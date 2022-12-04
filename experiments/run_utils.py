import os
import shutil
import datetime

import config

def _create_folders(model_name, dataset_name, tag_name):
    ''' Put logs of all jobs in the same jobarray into one directory '''

    current_time_est = datetime.datetime.now(datetime.timezone.utc)
    current_time = (current_time_est + datetime.timedelta(hours=config.timedelta)).strftime('%Y%m%d-%H%M%S')
    path = os.path.join(config.LOGS_PATH, model_name, f'{current_time}{dataset_name}{tag_name}')

    checkpoints_path = f'{path}/checkpoints'
    vis_path = f'{path}/visualizations'
    source_path = f'{path}/source'
    plots_path = f'{path}/plots'
    logs_path = f'{path}/logs'

    for f in [checkpoints_path, vis_path, source_path, plots_path, logs_path]:
        os.makedirs(f)

    return path, checkpoints_path, vis_path, source_path, plots_path, logs_path

def _log_model_source(target_dir, algorithm_name='all'):
    ''' Log raw model object file source -- copy it from the image to the data-server '''

    name_to_path = {
        'all': config.SOURCE_PATH,
        # 'prediction_attn': f'{pwd}/core/models/prediction/cnn_attention',
        # 'prediction_gnn': f'{pwd}/core/models/prediction/cnn_gnn',
    }

    model_path = name_to_path[algorithm_name]

    for root, dirs, files in os.walk(model_path):
        for f in files: 
            if f.endswith('.py'):
                file_src = os.path.join(root, f)
                file_trg = os.path.join(target_dir, f)
                shutil.copy(file_src, file_trg)
                os.chmod(file_trg, int('664', base=8))

def setup(model_name, dataset_name="", tag_name=""):
    path, checkpoints_path, vis_path, source_path, plots_path, logs_path = _create_folders(model_name, dataset_name, tag_name)
    _log_model_source(source_path)

    return path, checkpoints_path, vis_path, plots_path, logs_path