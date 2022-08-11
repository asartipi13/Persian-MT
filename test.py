import time
#from datasets import load_dataset
from IPython.display import display
# from IPython.html import widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import torch
from torch import optim
from torch.nn import functional as F
from transformers import AdamW, AutoModelForSeq2SeqLM, AutoTokenizer
import os, glob

# from torchtext.data.metrics import bleu_score
from torchmetrics.functional import sacre_bleu_score
# from torchmetrics import SacreBLEUScore

import wandb
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm_notebook
from tqdm import tqdm
sns.set()
import collections
import os

return_tensors='pt'

config_path = './server_config.json'
# config_path = './local_config.json'

with open(config_path) as sg:
    sg = json.load(sg)


wandb.login(key='e17de26c63319152119eefbc833bb10b2ceb7bfd')



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
sg['device'] = device
print(f'device: {device}')

model_repo = sg['model_repo']
model_dir = sg['model_dir']
best_model_path = sg['best_model_path']

data_dir = sg['data_dir']
train_path = data_dir + '/train.csv'
dev_path = data_dir + '/dev.csv'
test_path = data_dir + '/test.csv'

max_seq_len = sg['max_seq_len']
max_seq_test_len = sg['max_seq_test_len']
n_epochs = sg['n_epochs']

train_batch_size = sg['train_batch_size']
test_batch_size = sg['test_batch_size']
dev_batch_size = sg['dev_batch_size']
nrows = sg['nrows']

lr = sg['lr']
max_ngram = sg['max_ngram']
need_train = sg['need_train']

df_train = pd.read_csv(train_path) if nrows == -1 else pd.read_csv(train_path, nrows=nrows)
df_dev = pd.read_csv(dev_path) if nrows == -1 else pd.read_csv(dev_path, nrows=nrows)
df_test = pd.read_csv(test_path) if nrows == -1 else pd.read_csv(test_path, nrows=nrows)

Language_Token_Mapping = {
    df_train.columns[0]: "<" + df_train.columns[0] + ">",
    df_train.columns[1]: "<" + df_train.columns[1] + ">"
}

n_batches = int(np.ceil(len(df_train) / train_batch_size))
while ((n_batches-1) % 5 != 0):
    n_batches -= 1

print_freq = int((n_batches-1) / 5)
checkpoint_freq = int((n_batches-1) / 5)

total_steps = n_epochs * n_batches
n_warmup_steps = int(total_steps * 0.01)


def get_basic_config():
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    model = torch.load(best_model_path)
    model = model.to(device)
    special_token_dict = {"additional_special_tokens": list(Language_Token_Mapping.values())}
    tokenizer.add_special_tokens(special_token_dict)
    model.resize_token_embeddings(len(tokenizer))

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, n_warmup_steps, total_steps)

    return tokenizer, model, optimizer, scheduler


tokenizer, model, optimizer, scheduler = get_basic_config()


def encode_input_str(text, target_lang, tokenizer, seq_len, lang_token_map=Language_Token_Mapping):
  target_lang_token = Language_Token_Mapping[target_lang]
  input_ids = tokenizer.encode(
      text= target_lang_token + str(text),
      return_tensors=return_tensors,
      padding= 'max_length',
      truncation= True,
      max_length=seq_len
  )
  return input_ids[0]

def encode_target_str(text, tokenizer, seq_len, lang_token_map=Language_Token_Mapping):
  token_ids = tokenizer.encode(
      text=text,
      return_tensors=return_tensors,
      padding='max_length',
      truncation=True,
      max_length=seq_len
  )
  return token_ids[0]


def format_translation_data(translations, lang_token_map, tokenizer, seq_len, src, tar):
  # Choose a random 2 languages for in i/o
  # langs = list(lang_token_map.keys())
  input_lang, target_lang = [src, tar]
  #input_lang, target_lang = np.random.choice(langs, size=2, replace=False)
  # Get the translations for the batch
  input_text = translations[input_lang]
  target_text = translations[target_lang]
  
  if input_text is None or target_text is None:
    return None

  input_token_ids = encode_input_str(
      input_text, target_lang, tokenizer, seq_len, lang_token_map)
  
  target_token_ids = encode_target_str(
      target_text, tokenizer, seq_len, lang_token_map)

  return input_token_ids, target_token_ids


def transfrom_batch(batch, lang_token_map, tokenizer, seq_len, src, tar):
  inputs = []
  targets = []

  for translation_set in batch.iterrows():
    formatted_data = format_translation_data(
        translation_set[1], lang_token_map, tokenizer, seq_len, src, tar)
    
    if formatted_data is None:
      continue
    
    input_ids, target_ids = formatted_data
    inputs.append(input_ids.unsqueeze(0))
    targets.append(target_ids.unsqueeze(0))
    
#   batch_input_ids = torch.cat(inputs).cuda()
#   batch_target_ids = torch.cat(targets).cuda()

  batch_input_ids = torch.cat(inputs)
  batch_target_ids = torch.cat(targets)

  return batch_input_ids, batch_target_ids


def get_data_generator(dataset, lang_token_map, tokenizer, src, tar, batch_size=32, seq_len=128):
    dataset = dataset.sample(frac=1)
    for i in range(0, len(dataset), batch_size):
        raw_batch = dataset.iloc[i:i+batch_size,:]
        yield transfrom_batch(raw_batch, lang_token_map, tokenizer, seq_len, src, tar)
    # return raw_batch

def batch_encode_input_str(batch_text, target_lang, tokenizer, seq_len, lang_token_map=Language_Token_Mapping):
  target_lang_token = Language_Token_Mapping[target_lang]
  input_ids = tokenizer.batch_encode_plus(
      batch_text_or_text_pairs= [target_lang_token + str(text) for text in batch_text],
      return_tensors=return_tensors,
      padding= 'max_length',
      truncation= True,
      max_length=seq_len
  )
  return input_ids



def predict(df_test, model, src, tar):
    
    source = list(df_test[src].values)

    predicted = []
    print("len df test {}".format(len(df_test)))
    number_of_batches_in_test_df = int(len(df_test) / test_batch_size)

    for i in range(number_of_batches_in_test_df):
        batch = source[i * test_batch_size : i * test_batch_size + test_batch_size]
        inputs = batch_encode_input_str(
            batch_text = batch,
            target_lang = tar,
            tokenizer = tokenizer,
            seq_len = max_seq_test_len,
            lang_token_map = Language_Token_Mapping)

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

        output_tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=20, length_penalty=0.2, do_sample=False)

        preds = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        predicted.extend(preds)

    return predicted



def get_blue_score(df_test, predicted_target, max_n, tar):
    target = list(df_test[tar].values)[:len(predicted_target)]
    real_target = [[sent] for sent in target]
    bl_score = sacre_bleu_score(predicted_target, real_target, n_gram=max_n)
    return bl_score.item()

def test_process(src, tar, model_dir):
    predicted = predict(df_test, model, src, tar)
    #print(predicted)
    pd.DataFrame({"predicted": predicted}).to_csv(model_dir + '/predicted_{}_{}.csv'.format(src, tar))

    bl_score = get_blue_score(df_test, predicted, max_n=max_ngram, tar=tar)
    pd.DataFrame({"blue_score": [bl_score]}).to_csv(model_dir + '/bl_score_{}_{}.csv'.format(src, tar))

langs = list(Language_Token_Mapping.keys())


model_dir = "/".join(best_model_path.split("/")[:2])
src = best_model_path.split("/")[1].split("_")[-2]
tar = best_model_path.split("/")[1].split("_")[-3]

test_process(src, tar, model_dir)



