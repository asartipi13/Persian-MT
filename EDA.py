import json, hazm, stanza, os
from operator import index
from plotly import data
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True)

def extract_information(df, config):
    fa_tokens = []
    fa_length = []
    fa_ch_length = []

    en_crp = ''
    en_tokens = []
    en_length = []
    en_ch_length = []

    for index, row in df.iterrows():
        row['fa'] = str(row['fa'])
        tokens = hazm.word_tokenize(row['fa'])
        fa_tokens.extend(tokens)
        fa_length.append(len(tokens))
        fa_ch_length.append(len(row['fa']))

        row['en'] = str(row['en'])
        row['en'] += '\n\n'
        en_crp+= row['en']
        en_ch_length.append(len(row['en']))

        # if index == 50:
        #     break

    doc = nlp(en_crp)
    for i, sentence in enumerate(doc.sentences):
        en_length.append(len(sentence.tokens))
        [en_tokens.append(token.text) for token in sentence.tokens]

    # df = df[:51]
    df['fa_length'] = fa_length
    df['en_length'] = en_length
    
    df['fa_ch_length'] = fa_ch_length
    df['en_ch_length'] = en_ch_length

    df['fa_unique'] = [len(set(fa_tokens))] * len(df)
    df['en_unique'] = [len(set(en_tokens))] * len(df)

    df[['fa_length', 'en_length', 'fa_ch_length', 'en_ch_length', 'fa_unique', 'en_unique']].to_csv(config['output_directory'] + '/general.csv')

def data_gl_than(data, less_than=10, greater_than=0.0, col='fa_length'):
    data_length = data[col].values
    data_glt = sum([1 for length in data_length if greater_than < length <= less_than])
    data_glt_rate = (data_glt / len(data_length)) * 100
    # print(f'Texts with word length of greater than {greater_than} and less than {less_than} includes {data_glt_rate:.2f}% of the whole!')
    return data_glt_rate


def get_seq_len(df, col):
    minl = int(df[col].describe()[3])
    maxl = int(df[col].describe()[7])
    for less_than in range(minl,maxl):
        data_glt_rate = data_gl_than(data=df, less_than=less_than, col=col)
        if data_glt_rate >= 92:
            return less_than

def get_dataset_stat(df, config):

    info_word = {

        "avg_fa":int(df.describe()['fa_length'][1]),
        "min_fa":int(df.describe()['fa_length'][3]),
        "max_fa":int(df.describe()['fa_length'][7]),
        "92%_fa": int(get_seq_len(df, 'fa_length')),
        "all_fa": int(df['fa_length'].sum()),
        "unique_fa": int(df['fa_unique'][0]),

        "avg_en":int(df.describe()['en_length'][1]),
        "min_en":int(df.describe()['en_length'][3]),
        "max_en":int(df.describe()['en_length'][7]),
        "92%_en": int(get_seq_len(df, 'en_length')),
        "all_en": int(df['en_length'].sum()),
        "unique_en": int(df['en_unique'][0])

    }

    df_info_word = pd.DataFrame(info_word, index=[0])
    df_info_word.to_csv(config['output_directory'] + '/infor_word.csv', index=False)
    
    info_char = {

        "avgc_fa":int(df.describe()['fa_ch_length'][1]),
        "minc_fa":int(df.describe()['fa_ch_length'][3]),
        "maxc_fa":int(df.describe()['fa_ch_length'][7]),

        "avgc_en":int(df.describe()['en_ch_length'][1]),
        "minc_en":int(df.describe()['en_ch_length'][3]),
        "maxc_en":int(df.describe()['en_ch_length'][7]),

    }

    df_info_char = pd.DataFrame(info_char, index=[0])
    df_info_char.to_csv(config['output_directory'] + '/infor_char.csv', index=False)
    

def draw_charts(df, config):

    for col in df.columns:
        try:
            name = col + " distrobution"
            path = '{}/{}.png'.format(config['output_directory'], name)

            fig = go.Figure()
            fig = px.histogram(df, x=col)
            fig.update_layout(
                title_text=name,
                xaxis_title_text='Length',
                yaxis_title_text='Count',
                bargap=0.2,
                bargroupgap=0.2
            )
            fig.write_image(path, scale=2)

        except Exception as e:
            print('there is a error in chart drawing')
            raise e


if __name__ == '__main__':
    
    config_path = './Config/config_eda.json'

    with open(str(config_path), 'r+') as f:
        config = json.load(f)


    os.makedirs(config['output_directory'], exist_ok=True)

    df = pd.read_csv(config['file_path'])
    extract_information(df, config)

    df = pd.read_csv(config['output_directory'] + '/general.csv')
    get_dataset_stat(df, config)
    draw_charts(df[['fa_length', 'en_length', 'fa_ch_length', 'en_ch_length']], config)





    




    

