import json, os
from plotly import data
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.types import *



spark = sparknlp.start()


def get_pipeline(col):

    documentAssembler = DocumentAssembler() \
        .setInputCol(col) \
        .setOutputCol("document")

    sentenceDetector = SentenceDetector() \
        .setInputCols(["document"]) \
        .setOutputCol("sentence")

    regexTokenizer = Tokenizer() \
        .setInputCols(["sentence"]) \
        .setOutputCol("token")

    finisher = Finisher() \
        .setInputCols(["token"]) \
        .setIncludeMetadata(True)
    
    pipeline = Pipeline().setStages([
    documentAssembler,
    sentenceDetector,
    regexTokenizer,
    finisher
    ])
    return pipeline

def extract_information(df, config):

    fa_pip = get_pipeline('fa')
    en_pip = get_pipeline('en')
    
    fa = fa_pip.fit(df).transform(df).select("finished_token", 'fa')
    en = en_pip.fit(df).transform(df).select("finished_token", 'en')


    fa_tokens = []
    fa_length = []
    fa_ch_length = []

    en_tokens = []
    en_length = []
    en_ch_length = []

    result_fa = fa.collect()

    for item in result_fa:
        fa_tokens.extend(item.finished_token)
        fa_length.append(len(item.finished_token))
        fa_ch_length.append(len(item.fa))
    
    del result_fa
    del fa

    result_en = en.collect()

    for item in result_en:
        en_tokens.extend(item.finished_token)
        en_length.append(len(item.finished_token))
        en_ch_length.append(len(item.en))
   
    del result_en
    del en

    df_general = pd.DataFrame({
        "fa_length": fa_length,
        "en_length": en_length,
        "fa_ch_length": fa_ch_length,
        "en_ch_length":en_ch_length,
        "fa_unique": [len(set(fa_tokens))] * df.count(),
        "en_unique": [len(set(en_tokens))] * df.count()
    })

    df_general.to_csv(config['output_directory'] + '/general.csv', index=False)

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
    return -1

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
            name = col + "_distrobution"
            path = '{}/{}.png'.format(config['output_directory'], name)
            df = df[df[col]<200]
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
    datasets = ['Mizan', 'PEPC_Bidirectional', 'PEPC_Onedirectional', 'TEP', 'TEP++', 'OpenSubtitles', 'Bible', 'Quran', 'ParsiNLU']

    # schema = StructType([StructField("fa", StringType(), True), StructField("en", StringType(), True)])

    # for d in datasets:
    #     config_path = './Config/config_eda.json'

    # with open(str(config_path), 'r+') as f:
    #     config = json.load(f)

    for d in datasets:
        config = {"file_path":"", "output_directory":""}
        config['file_path'] = './drive/MyDrive/data/{}/en-fa.csv'.format(d)
        config['output_directory'] = './data/{}/eda'.format(d)

        print(d)

        # os.makedirs(config['output_directory'], exist_ok=True)

        # df = pd.read_csv(config['file_path'])
        # df = spark.createDataFrame(df, schema=schema)

        # extract_information(df, config)

        df = pd.read_csv(config['output_directory'] + '/general.csv')
        # get_dataset_stat(df, config)
        draw_charts(df[['fa_length', 'en_length', 'fa_ch_length', 'en_ch_length']], config)
        print("finish {}".format(d))







    

