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
    model = None
    # model_repo
    if need_train:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_repo)
    else:
        model = torch.load(model_dir+ '/model.pt')

    # tokenizer.save_pretrained('./model-and-tokenizer/')
    # model.save_pretrained('./model-and-tokenizer/')

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


def eval_model(model, gdataset, tokenizer, batch_size, src, tar, max_iters=8):
    test_generator = get_data_generator(gdataset, Language_Token_Mapping, tokenizer, src, tar, batch_size, seq_len=max_seq_len)
    eval_losses = []
    for i, (input_batch, label_batch) in enumerate(test_generator):
        if i >= max_iters:
            break

        input_batch = input_batch.to(device)
        label_batch = label_batch.to(device)
        
        model_out = model.forward(
            input_ids = input_batch,
            labels = label_batch)
        eval_losses.append(model_out.loss.item())

    return np.mean(eval_losses)



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
    return bl_score


def train_process(src, tar, model_dir, logger):
    history = collections.defaultdict(list)
    history_tota = collections.defaultdict(list)

    total_train_losss = []
    total_dev_losss = []

    best_val_loss = np.Infinity
    # best_model = deepcopy(model)
    bl_scores = []
    if need_train:

        try:
            for epoch_idx in range(n_epochs):
                train_losses = []
                dev_losses = []
                # Randomize data order
                data_generator = get_data_generator(df_train, Language_Token_Mapping, tokenizer, src, tar, train_batch_size, max_seq_len)
                start_time = time.time()            
                for batch_idx, (input_batch, label_batch) in tqdm(enumerate(data_generator), total=n_batches):
                # for batch_idx, (input_batch, label_batch) in enumerate(data_generator):

                    optimizer.zero_grad()
                    input_batch = input_batch.to(device)
                    label_batch = label_batch.to(device)
                    # Forward pass
                    model_out = model.forward(input_ids = input_batch, labels = label_batch)

                    # Calculate loss and update weights
                    loss = model_out.loss
                    train_losses.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    # Print training update info
                    if (batch_idx) % print_freq == 0:
                        avg_loss = np.mean(train_losses[-print_freq:])
                        print('Epoch: {} | Step: {} | Avg. loss: {:.3f} | lr: {}'.format(
                            epoch_idx+1, batch_idx+1, avg_loss, scheduler.get_last_lr()[0]))
                        
                        # logger.log({"certain_steps_train_loss": float(avg_loss)})

                    
                    if (batch_idx) % checkpoint_freq == 0:
                        dev_loss = eval_model(model, df_dev, tokenizer, dev_batch_size, src, tar, max_iters=16)
                        # print('Saving model with test loss of {:.3f}'.format(dev_loss))
                        dev_losses.append(dev_loss)
                        print('Epoch: {} | Step: {} | Dev. loss: {:.3f} | lr: {}'.format(
                            epoch_idx+1, batch_idx+1, dev_loss, scheduler.get_last_lr()[0]))
                        # torch.save(model, model_dir + '/model.pt')

                        # logger.log({"certain_steps_val_loss": float(dev_loss)})

                epoch_train_loss = np.mean(train_losses)
                epoch_dev_loss = np.mean(dev_losses)

                if best_val_loss > epoch_dev_loss:
                    best_val_loss = epoch_dev_loss
                    for filename in glob.glob(model_dir + '/best_model*'):
                        os.remove(filename) 
                    torch.save(model, model_dir + '/best_model_{}.pt'.format(str(epoch_idx+1)))

                total_train_losss.extend(train_losses)
                total_dev_losss.extend(dev_losses)

                # logger.log({"train_loss": float(epoch_train_loss),
                #         "val_loss": float(epoch_dev_loss)})

                history['train_loss'].append(epoch_train_loss)
                history['val_loss'].append(epoch_dev_loss)
                history['time'].append(time.time() - start_time)
                
                # blue score for each epoch
                predicted = predict(df_test, model, src, tar)
                bl_score = get_blue_score(df_test, predicted, max_n=max_ngram, tar=tar)
                bl_scores.append(bl_score)
 
            torch.save(model, model_dir + '/last_model.pt')

            pd.DataFrame({"total_train_losss": total_train_losss}).to_csv(model_dir + '/total_train_losss.csv')
            pd.DataFrame({"total_dev_losss": total_dev_losss}).to_csv(model_dir + '/total_dev_losss.csv')
            pd.DataFrame(history).to_csv(model_dir + '/history.csv')
            pd.DataFrame({"blue_score": bl_scores}).to_csv(model_dir + '/each_bl_score.csv')


        except Exception as e:
            with open(model_dir+'/error.txt', 'w') as f:
                f.write(str(e))

            torch.save(model, model_dir + '/last_model.pt')
            
            pd.DataFrame({"total_train_losss": total_train_losss}).to_csv(model_dir + '/total_train_losss.csv')
            pd.DataFrame({"total_dev_losss": total_dev_losss}).to_csv(model_dir + '/total_dev_losss.csv')
            pd.DataFrame(history).to_csv(model_dir + '/history.csv')

def test_process(src, tar, model_dir):
    predicted = predict(df_test, model, src, tar)
    #print(predicted)
    pd.DataFrame({"predicted": predicted}).to_csv(model_dir + '/predicted_{}_{}.csv'.format(src, tar))

    bl_score = get_blue_score(df_test, predicted, max_n=max_ngram, tar=tar)
    pd.DataFrame({"blue_score": [bl_score]}).to_csv(model_dir + '/bl_score_{}_{}.csv'.format(src, tar))

langs = list(Language_Token_Mapping.keys())
for i in range(1):


    source_lang = langs[i]
    target_lang = langs[1-i]

    if source_lang=='en':
        source_lang = target_lang
        target_lang = 'en'

    new_model_dir = "{}/{}_{}_{}_{}_{}".format(model_dir, model_repo.replace("/", "-"), data_dir.split('/')[-1], source_lang, target_lang ,n_epochs)
    # new_model_dir = "{}/{}_{}_{}_{}_{}".format(model_dir, model_repo.replace("/", "-"), "OS_", source_lang, target_lang ,n_epochs)
    
    os.makedirs(new_model_dir, exist_ok=True)
    # wandb.init(project=new_model_dir.split('/')[-1], entity="persian-mt", dir=new_model_dir)

    if need_train:
        train_process(source_lang, target_lang, new_model_dir, None)

    test_process(source_lang, target_lang, new_model_dir)


