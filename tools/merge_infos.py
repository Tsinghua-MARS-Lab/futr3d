import mmcv
import ipdb
train_path = 'data/nuscenes_5sweeps_infos_train_radar.pkl'
val_path = 'data/nuscenes_5sweeps_infos_val_radar.pkl'

train_infos = mmcv.load(train_path)
val_infos = mmcv.load(val_path)

trainval_infos = {}
trainval_infos['metadata'] = train_infos['metadata']
trainval_infos['infos'] = train_infos['infos'] + val_infos['infos']

mmcv.dump(trainval_infos, 'data/nusc_new/nuscenes_5sweeps_infos_trainval_radar.pkl')
print(len(trainval_infos['infos']))
print(len(train_infos['infos']))
print(len(val_infos['infos']))