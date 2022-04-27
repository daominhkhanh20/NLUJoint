import argparse

import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path_folder_data', type=str, default='assets/data/bkai')
parser.add_argument('--mode', type=str, default='dev')
args = parser.parse_args()
data = pd.read_csv(f"{args.path_folder_data}/{args.mode}/{args.mode}.csv")
all_sentences = data['text'].values.tolist()
all_intents = data['intent'].values.tolist()
all_slots = data['tag'].values.tolist()

relabel_intents = []

decrease_words = ['giảm', 'hạ', 'xuống']
increase_words = ['tăng', 'thêm']
level_words = ['mức']
check_words = ['kiểm tra', 'kiểm soát', 'quét', 'mở hay đóng', 'đóng hay mở', 'bật hay tắt', 'tắt hay bật', 'check',
               'bật hoặc tắt', 'tắt hoặc bật', 'còn bật', 'mở ah', 'đã đóng', 'đã tắt', 'chưa đóng', ' được mở',
               'mở hoặc đóng', 'đóng hoặc mở', 'đang tắt', 'bật ah', 'đóng ah', 'được tắt']
on_off_words = ['bật', 'mở', 'tắt', 'đóng']
set_words = ['cài đặt', 'đặt']
color_words = ['xanh', 'đỏ', 'tím', 'vàng', 'lục', 'lam', 'chàm', 'hồng', 'tía', 'trắng', 'đen', 'cam', 'màu']

is_diff = []


def is_true(sent, list_word):
    word_split = sent.split(" ")
    for word in list_word:
        if (len(word.split(" ")) == 1 and word in word_split) or (len(word.split(" ")) > 1 and word in sent):
            return True
    return False


for sentence, intent in tqdm(zip(all_sentences, all_intents), total=len(all_sentences)):
    is_percent_exist = 'phần trăm' in sentence
    if not is_true(sentence, on_off_words) and is_true(sentence,
                                                       decrease_words + increase_words) and not is_percent_exist:
        if intent != 'smart.home.decrease.level' and is_true(sentence, decrease_words):
            relabel_intents.append('smart.home.decrease.level')
        elif intent != 'smart.home.increase.level' and is_true(sentence, increase_words) and not is_true(sentence, decrease_words):
            relabel_intents.append('smart.home.increase.level')
        else:
            relabel_intents.append(intent)
    elif is_true(sentence, check_words):
        if is_true(sentence, on_off_words) and intent == 'smart.home.set.level':
            relabel_intents.append(intent)
        else:
            if intent != 'smart.home.check.status':
                relabel_intents.append('smart.home.check.status')
            else:
                relabel_intents.append(intent)
    elif is_true(sentence, set_words) and not is_true(sentence, color_words) and not is_percent_exist:
        if intent != 'smart.home.set.level':
            relabel_intents.append('smart.home.set.level')
        else:
            relabel_intents.append(intent)
    elif is_true(sentence, on_off_words) and not is_true(sentence, level_words) and not is_true(sentence, increase_words) \
            and not is_true(sentence, color_words) and not is_percent_exist:
        if intent != 'smart.home.device.onoff':
            relabel_intents.append('smart.home.device.onoff')
        else:
            relabel_intents.append(intent)
    else:
        relabel_intents.append(intent)

    if intent != relabel_intents[-1]:
        is_diff.append(True)
    else:
        is_diff.append(0)

df = pd.DataFrame({'text': all_sentences, 'is_diff': is_diff, 'intent': all_intents, 'relabel_intent': relabel_intents,
                   'tag': all_slots})
df.to_csv(f"{args.path_folder_data}/{args.mode}/relabel_{args.mode}.csv", index=False)
