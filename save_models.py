from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

# mT5-Small (300 million parameters): gs://t5-data/pretrained_models/mt5/small
# mT5-Base (580 million parameters): gs://t5-data/pretrained_models/mt5/base
# mT5-Large (1.2 billion parameters): gs://t5-data/pretrained_models/mt5/large
# mT5-XL (3.7 billion parameters): gs://t5-data/pretrained_models/mt5/xl
# mT5-XXL (13 billion parameters): gs://t5-data/pretrained_models/mt5/xxl

models = ['google/mt5-small', 'google/mt5-base', 'google/mt5-large', 'google/mt5-xl', 'google/mt5-xxl']
models = [ 'facebook/nllb-200-distilled-600M', 'facebook/nllb-200-distilled-1.3B','facebook/nllb-200-1.3B', 'facebook/nllb-200-3.3B']

models = ['facebook/nllb-200-distilled-600M']

for m in models:
  tokenizer = AutoTokenizer.from_pretrained(m)
  model = AutoModelForSeq2SeqLM.from_pretrained(m)
  os.makedirs('./{}'.format(m), exist_ok=True)

  tokenizer.save_pretrained('./{}'.format(m))
  model.save_pretrained('./{}'.format(m))