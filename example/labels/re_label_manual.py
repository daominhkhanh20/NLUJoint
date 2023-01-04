import argparse

import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path_folder_data', type=str, default='assets/data/bkai')
parser.add_argument('--mode', type=str, default='dev')
args = parser.parse_args()
data = pd.read_csv(f"{args.path_folder_data}/{args.mode}/{args.mode}.csv")
sents = data['text'].values.tolist()
intent_labels = data['intent'].values.tolist()
slot_labels = data['tag'].values.tolist()

def deduplicate_data(sents, intents, slots):
    print('deduplicate')
    new_inputs, new_intents, new_slots = [], [], []
    set_sents = list()
    for sent, intent, slot in zip(sents, intents, slots):
        if sent in set_sents:
            continue
        set_sents.append(sent)
        new_inputs.append(sent)
        new_intents.append(intent)
        new_slots.append(slot)
    if len(sents) != len(new_inputs):
        print(f"Before:{len(input)} - Now:{len(new_inputs)}")
    return new_inputs, new_intents, new_slots


def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False
    
def fixed_label():
    print("Before: Number: ", len(sents))
    older_intents, older_slots = [], []
    new_sents, new_slot_labels, new_labels, is_diff = [], [], [], []
    for sent, slot, label in zip(sents, slot_labels, intent_labels):
        # fix intent
        new_label = None
        if (
            "thay đổi" in sent
            and not any(x in sent for x in ["tăng", "giảm", "màu", "xám", "xanh", "trắng", "tím"])
            and "set.level" not in label
        ):
            new_label = "smart.home.set.level"
        elif not any(x in sent for x in ["phần"]) and "bật" in sent and "percentage" in label:
            new_label = "smart.home.device.onoff"
        elif not any(x in sent for x in ["phần"]) and "giảm" in sent and "percentage" in label:
            new_label = "smart.home.decrease.level"
        elif not any(x in sent for x in ["phần"]) and "tăng" in sent and "percentage" in label:
            new_label = "smart.home.increase.level"
        elif (
            "set.level" in label
            and any(x in sent for x in ["bật"])
            and not any(x in sent for x in ["cấp", "mức", "giảm", "tăng"])
        ):
            new_label = "smart.home.device.onoff"
        elif "set.percentage" in label and "phần" not in sent:
            if any(x in sent for x in ["đóng", "bật", "mở"]):
                new_label = "smart.home.device.onoff"
            elif "vui vẻ" in sent:
                new_label = "greeting"
            else:
                new_label = "smart.home.set.level"
        elif "percentage" in label and len(sent.split()) < 5 and "phần" not in sent:
            new_label = "smart.home.set.level"
        else:
            new_label = label

        # fix slot
        if "xuống" in sent or "tăng" in sent or "giảm" in sent:
            if "final-valuesyspercentage" in slot:
                slot = slot.replace("final-valuesyspercentage", "change-valuesyspercentage")

        slot = slot.split()
        # if 'device' in short:
        new_slot = []
        if (
            "bóng" in sent
            and "set.color" in new_label
            and not slot[sent.split().index("bóng")] != "O"
        ):
            idx = sent.split().index("bóng")
            assert slot[idx + 1] == "B-devicedevice"
            for i in range(len(slot)):
                if i == idx:
                    new_slot.append("B-devicedevice")
                elif i == idx + 1:
                    new_slot.append("I-devicedevice")
                else:
                    new_slot.append(slot[i])
        elif (
            "đèn bóng trần" in sent
            and "decrease.level" in new_label
            and slot[sent.split().index("đèn")] == "O"
        ):
            idx = sent.split().index("đèn")
            assert slot[idx : idx + 3] == ["O"] * 3
            for i in range(len(slot)):
                if i == idx:
                    new_slot.append("B-devicedevice")
                elif i == idx + 1 or i == idx + 2:
                    new_slot.append("I-devicedevice")
                else:
                    new_slot.append(slot[i])
        else:
            new_slot = [x for x in slot]

        sent = sent.split()
        for i in range(len(new_slot)):
            if i > 0 and (
                (i == len(new_slot) - 1 and is_float(sent[i]) and is_float(sent[i - 1]))
                or (
                    i < len(new_slot) - 1
                    and is_float(sent[i])
                    and is_float(sent[i - 1])
                    and sent[i + 1] == "phần"
                )
            ):
                if new_slot[i] == "sysnumbersysnumber" and new_slot[i - 1] == "roomroom":
                    new_slot[i] = "floornumberfloornumber"

        sent = " ".join(sent)
        new_slot = " ".join(new_slot)

        if new_label in [
            "smart.home.decrease.percentage",
            "smart.home.set.percentage",
            "smart.home.increase.percentage",
        ]:
            if "percentage" not in new_slot:
                continue

        new_sents.append(sent)
        new_slot_labels.append(new_slot)
        new_labels.append(new_label)
        older_intents.append(label)
        older_slots.append(" ".join(slot))
        if new_slot != " ".join(slot):
            is_diff.append(2)
        elif label != new_label:
            is_diff.append(1)
        else:
            is_diff.append(0)
    new_sents, new_slot_labels, new_labels = deduplicate_data(new_sents, new_slot_labels, new_labels)
    return new_sents, new_slot_labels, new_labels, is_diff, older_slots, older_intents

def fix_label2():
    relabel_intents, is_confuse = [], []

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


    for sentence, intent in tqdm(zip(sents, intent_labels), total=len(sents)):
        is_percent_exist = 'phần trăm' in sentence
        if not is_percent_exist and 'percentage' in intent:
            is_confuse.append(1111111111111111)
        else:
            is_confuse.append(0)
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
        if is_diff[-1] == 0 and is_confuse[-1] == 1111111111111111:
            if "thay đổi" in sentence:
                relabel_intents[-1] = "smart.home.set.level" 
                is_diff[-1] = True
            print(sentence, '---',intent)
    return sents, slot_labels, relabel_intents, is_diff, slot_labels, intent_labels, is_confuse
            
            

sents, all_slots, relabel_intents, is_diff, older_slots, older_intents, is_confuse = fix_label2()
df = pd.DataFrame({'text': sents, 'is_diff': is_diff, 'relabel_intent': relabel_intents,'older_intents': older_intents, 'is_confuse': is_confuse,'tag': all_slots, 'orlder_tag': older_slots})
df.to_csv(f"{args.path_folder_data}/{args.mode}/relabel_{args.mode}.csv", index=False)
