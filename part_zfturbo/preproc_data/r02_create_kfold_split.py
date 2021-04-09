# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a00_common_functions import *
from sklearn.model_selection import StratifiedKFold, KFold


def create_kfold_split_uniform(folds, seed=42):
    cache_path = OUTPUT_PATH + 'kfold_split_{}_{}.csv'.format(folds, seed)
    if not os.path.isfile(cache_path):
        kf = KFold(n_splits=folds, random_state=seed, shuffle=True)
        train = pd.read_csv(INPUT_PATH + 'train.csv')
        image_ids = train['image_id'].unique()
        print(len(image_ids))

        s = pd.DataFrame(image_ids, columns=['image_id'])
        s['fold'] = -1
        for i, (train_index, test_index) in enumerate(kf.split(s.index)):
            s.loc[test_index, 'fold'] = i
        s.to_csv(OUTPUT_PATH + 'kfold_split_{}_{}.csv'.format(folds, seed), index=False)
        print('No folds: {}'.format(len(s[s['fold'] == -1])))

        for i in range(folds):
            part = s[s['fold'] == i]
            print(i, len(part))
    else:
        print('File already exists: {}'.format(cache_path))


def create_kfold_split_external_data(folds, seed=42):
    cache_path = OUTPUT_PATH + 'kfold_split_external_{}_{}.csv'.format(folds, seed)
    if not os.path.isfile(cache_path):
        kf = KFold(n_splits=folds, random_state=seed, shuffle=True)
        train = pd.read_csv(INPUT_PATH + 'nih.csv')
        image_ids = train['image_id'].unique()
        print(len(image_ids))

        s = pd.DataFrame(image_ids, columns=['image_id'])
        s['fold'] = -1
        for i, (train_index, test_index) in enumerate(kf.split(s.index)):
            s.loc[test_index, 'fold'] = i
        s.to_csv(cache_path, index=False)
        print('No folds: {}'.format(len(s[s['fold'] == -1])))

        for i in range(folds):
            part = s[s['fold'] == i]
            print(i, len(part))
    else:
        print('File already exists: {}'.format(cache_path))


def create_kfold_split_siim_data(folds, seed=42):
    cache_path = OUTPUT_PATH + 'kfold_split_siim_{}_{}.csv'.format(folds, seed)
    if not os.path.isfile(cache_path):
        kf = KFold(n_splits=folds, random_state=seed, shuffle=True)
        train = pd.read_csv(INPUT_PATH + 'siim_pneumothorax.csv')
        image_ids = train['image_id'].unique()
        print(len(image_ids))

        s = pd.DataFrame(image_ids, columns=['image_id'])
        s['fold'] = -1
        for i, (train_index, test_index) in enumerate(kf.split(s.index)):
            s.loc[test_index, 'fold'] = i
        s.to_csv(cache_path, index=False)
        print('No folds: {}'.format(len(s[s['fold'] == -1])))

        for i in range(folds):
            part = s[s['fold'] == i]
            print(i, len(part))
    else:
        print('File already exists: {}'.format(cache_path))


def merge_splits(files):
    res = []
    for f in files:
        s = pd.read_csv(f)
        res.append(s)
    res = pd.concat(res, axis=0)
    res.to_csv(OUTPUT_PATH + 'kfold_split_with_nih_siim_5_42.csv', index=False)
    return res


if __name__ == '__main__':
    create_kfold_split_uniform(5, seed=42)
    if 0:
        create_kfold_split_external_data(5, seed=42)
        create_kfold_split_siim_data(5, seed=42)
        merge_splits([
            OUTPUT_PATH + 'kfold_split_5_42.csv',
            OUTPUT_PATH + 'kfold_split_external_5_42.csv',
            OUTPUT_PATH + 'kfold_split_siim_5_42.csv'
        ])