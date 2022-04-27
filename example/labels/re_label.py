import argparse
from ast import Raise
from nlu_transformer.module.assign_label import RelabelByCluster, RelabelByML

parser = argparse.ArgumentParser()
parser.add_argument('--path_folder_data', type=str, default='assets/data/bkai')
parser.add_argument('--path_save_data', type=str, default='assets/data/bkai_relabel')
parser.add_argument('--kernel', type=str,  default='linear')
parser.add_argument('--continuous_replace', type=str, default=False)
parser.add_argument('--mode_relabel', type=str, required=True)
parser.add_argument('--top_k', type=int, default=20)
parser.add_argument('--add_name', type=str, default=None)
parser.add_argument('--sent', type=str, default=None)
args = parser.parse_args()


if args.mode_relabel == 'cluster':
    relabel_model = RelabelByCluster(path_folder_data=args.path_folder_data,
                            path_save_data=args.path_save_data,
                            top_k_common=args.top_k)

    relabel_model.relabel_by_word_counter(sent=args.sent)
elif args.mode_relabel == 'ml':
    if args.add_name is not None:
        path_save = f"{args.path_save_data}_{args.kernel}_{args.add_name}"
    else:
        path_save = f"{args.path_save_data}_{args.kernel}"
    relabel_model = RelabelByML(
        path_folder_data=args.path_folder_data,
        path_save_data=path_save,
        text_column_name='filter_text',
        continuous_replace=args.continuous_replace,
        kernel=args.kernel
    )
    relabel_model.relabel()
    # print(relabel_model.fit_model('mình cần tăng thiết bị lên phần trăm với'))
else:
    raise Exception(f"{args.mode_relabel} isn't support")