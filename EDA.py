from re import template
import pandas as pd
import nltk
import collections as co
from bidi.algorithm import get_display
import arabic_reshaper
from wordcloud_fa import WordCloudFa
import os
import hazm
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
import preprocess
import argparse
import json
import sys

class error_handler():

    class error(Exception):
        def __init__(self, message):
            super().__init__(message)

    def instance(object, object_name, type):
        # check if parameter object is instance of parameter type
        if isinstance(object, type):
            return '1'
        message = f"{object_name} is invalid! {object_name} is {object}, but it should be {type}"
        
        raise error_handler.error(message)
    
    def column_name_validation(column_name, column_variable_name, dataframe, dataframe_variable_name):

        # checks if type of column name and dataframe is valid
        error_handler.instance(column_name,column_variable_name, str)
        error_handler.instance(dataframe, dataframe_variable_name, pd.DataFrame)

        #checks if column name exists in dataframe columns
        if column_name in dataframe.columns:
            return '1'

        message = f"{column_variable_name} is invalid! There is no {column_name} column in the {dataframe_variable_name}"
        
        raise error_handler.error(message)
    
    def directory(directory, directory_name):

        # checks if directory have valid type
        error_handler.instance(directory, directory_name, str)

        # checks if output directory exists
        if os.path.isdir(directory):
            return '1'

        message = f"\"{directory_name}\" is invalid! Directory \"{directory}\" doesn't exist!"
        
        raise error_handler.error(message)

    def file(output_directory, name, extension, mode=1):
        # it take care no data is removed or overwritten
        # in the case output file name exists it creates new file names
        # mode parameter controls if extension of a file should be returned or not
        # so that in mode 1 path is returned with the ile extension
        # in mode 0 path is returned without the file extension

        path = os.path.join(output_directory, name)
        i = 0
        while os.path.isfile(path+extension):
            path = os.path.join(output_directory, name+str(i))
            i+=1
        if mode:
            path+=extension
        return path

     
class eda:

    ####### methods #############################################################

    def __init__(
        self,
        file_path,
        output_directory,
        comment_column_name,
        preproccess_config=False
    ):
               
        # config class attributes
        self.output_directory = output_directory
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        if preproccess_config and os.path.isfile(preproccess_config):
            with open(preproccess_config, 'r+') as f:
                preproccess_config = json.load(f)
            preprocess.DataProcessor(**preproccess_config)
            file_path = preproccess_config['output_directory']
            comment_column_name = 'comment'

        self.dataframe = pd.read_csv(file_path, encoding='utf-8').dropna(subset=[comment_column_name])

        self.comment_column_name = comment_column_name
        self.word_tokens = []
        
        #because the frequeny tokenize use hazm.word_tokenizer in order to get words, there is no need for any other tokenizer for word frequency plot
        a = lambda x:x
        # config tqdm in order to create progresss bar
        log = tqdm(total=0, position=0, bar_format='{desc}')
        progressbar = tqdm(total=6, position=1, desc='eda')
        # dict of all class functions as dict keys and their arguments as dict values
        driver_function_dict = [
            [eda.word_cloud, []],
            [eda.comment_length_destribution, []],
            [eda.word_length_destribution, []],
            [eda.token_frequency, [a, 'word']],
            [eda.token_frequency, [nltk.ngrams, 'bigrams' , {'n':2}]],
            [eda.token_frequency, [nltk.ngrams, 'trigrams' , {'n':3}]],
        ]

        for f, args in driver_function_dict:
            f(self , *args)
            log.set_description_str('Current: '+str(self.output_directory.split('\\')[-3:]))
            progressbar.update(1)

 
        # final cleaning of  progressbar and its log
        log.reset()
        log.set_description_str('Finished!')
        log.close()
        progressbar.update(6)
        progressbar.close()


    def fix_persian_text_misorder(x):
        # some time word orders change when you want to use them in matplotlib or opencv. It needed to be reshaped.
        return get_display(arabic_reshaper.reshape(x))

    def word_cloud(self):

        try:
            # create a text from all comments
            text = self.dataframe[self.comment_column_name].to_string(index = False, header = False).replace('\n', ' ') 

            # wordcloud config
            wc = WordCloudFa(width=1200, height=800, persian_normalize=True, include_numbers=False, collocations=False,background_color='white')
            
            # generate wordcloud
            wc = wc.generate(text)
            
            # save generated image as png file in output folder
            wc = wc.to_image()
            path = error_handler.file(self.output_directory, 'wordcloud', '.png')
            wc.save(path)
                
        except Exception as e:
            print('There is a problem in word_cloud!')
            raise e

    def comment_length_destribution(self):

        try:

            name = 'comment length destribution'
            path = error_handler.file(self.output_directory, name, '.png')

            # compute length of a comment
            self.dataframe['comment length'] = self.dataframe[self.comment_column_name].apply(lambda x: len(x))

            # create histogram plot and config it
            fig = go.Figure()
            fig = px.histogram(self.dataframe, x='comment length')
            #fig.update_xaxes(range=[self.dataframe['comment length'].min(), self.dataframe['comment length'].min()])
            fig.update_layout(
                title_text=name,
                xaxis_title_text='Comment Character-Level Length',
                yaxis_title_text='Frequency',
                bargap=0.2,
                bargroupgap=0.2
            )

            # save histogram
            fig.write_image(path, scale=2)

        except Exception as e:
            print('there is a error in comment_length_destribution')
            raise e
    
    def word_length_destribution(self):

        try:

            name = 'word length destribution'
            path = error_handler.file(self.output_directory, name, '.png')

            # compute words
            if not bool(self.word_tokens):
                self.dataframe[self.comment_column_name].apply(lambda x : self.word_tokens.extend(hazm.word_tokenize(x)))
            
            # compute words length
            words_length = list(map(len, self.word_tokens))
            
            ####### temp ########################################################
            #text = self.dataframe[self.comment_column_name].to_string(index = False, header = False).replace('\n', ' ') 
            #words_length = list(map(len, text.split()))
            #df = pd.DataFrame({'word':text.split(),'word length':words_length})
            #df = df.loc[df['word length'] > 10]
            #df.to_excel(error_handler.file(self.output_directory, name, '.xlsx'))
            #####################################################################

            # create historam plot and config it
            fig = go.Figure()
            fig = px.histogram(x=words_length)
            fig.update_layout(
                title_text=name,
                xaxis_title_text='Word Length Destribution',
                yaxis_title_text='Frequency',
                bargap=0.2,
                bargroupgap=0.2
            )

            # save histogram
            fig.write_image(path, scale=2)

        except Exception as e:
            print('there is a error in comment_length_destribution')
            raise e

    def token_frequency(self, tokenizer=None, token_name='word', args = dict()):
        
        try:

            name = f'{token_name.title()} Frequency'
            path = error_handler.file(self.output_directory, name, '.png')

            # in order to lessen compution it check if wordtokens computed before or not
            if not bool(self.word_tokens):
                self.dataframe[self.comment_column_name].apply(lambda x : self.word_tokens.extend(hazm.word_tokenize(x)))
            
            # compute tokens
            tokens = []
            tokens.extend(tokenizer(self.word_tokens, **args))
            
            tokens = list(map(str, tokens))

            # count tokens and get most common ones
            tokens = co.Counter(tokens).most_common(30)      
            tokens, count = zip(*tokens)

            # create bar plot and config it
            fig = go.Figure()
            fig = px.bar(y=tokens, x=count, orientation='h')
            fig.update_layout(
                title_text=name,
                xaxis_title_text='Count',
                yaxis_title_text=token_name.title(),
                bargap=0.2,
                bargroupgap=0.2
            )
            fig.update_yaxes(
                dtick=1
            )

            # save bar plot
            fig.write_image(path, scale = 2)

        except Exception as e:
            print('there is a error in token_frequency')
            raise e

    def generate_config_template(output_directory):
        
        template={
            'file_path':'path_to_your_dataset',
            'output_directory': 'output_directory_path_for_generated_files',
            'comment_column_name': 'comment_column_name_in_your_dataset',
            'preproccess_config':'if_you_want_to_do_preprocess_pass_path_of_preproccess_class_config_json_file_to_it_otherwise_do_not_pass_it'
        }
        
        path = error_handler.file(output_directory, 'config_eda', '.json')
        with open(path, 'w') as f:
            json.dump(template, f, indent=1)
    #############################################################################




####################################### driver code ############################################

if __name__ == '__main__':
    
    config_path = './Config/config_eda.json'


    with open(str(config_path), 'r+') as f:
        config = json.load(f)
    
    e = eda(**config)
 
