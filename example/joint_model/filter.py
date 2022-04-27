import os
import argparse

from nlu_transformer.module.filter_data import FilterData

parser = argparse.ArgumentParser()
parser.add_argument('--path_folder_data', type=str, default='assets/data/bkai')
parser.add_argument('--has_relabel', default=False, type=lambda x: x.lower() == 'true')
parser.add_argument('--path_save_data', type=str, default='assets/data/bkai_filter')

args = parser.parse_args()

if os.path.exists(args.path_save_data):
    os.system(f"rm -r {args.path_save_data}")

filter = FilterData(path_folder_data=args.path_folder_data, has_relabel=args.has_relabel)

filter.filter_correct_data(path_save_data=args.path_save_data)
