import random
import argparse
import pandas as pd
from random import shuffle
import re

#stop words list
stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', ''}

def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

def augment_caption(sentence):

    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = list(filter(lambda x: x not in stop_words, words))
    shuffle(words)
    sentence = ' '.join(words)
    return sentence

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type = str, default = None, help = "Path to train data csv/tsv file")
    parser.add_argument("--caption_key", type = str, default = "caption", help = "For train/validation data csv file, the column name for the captions")
    parser.add_argument("--delimiter", type = str, default = ",", help = "For train/validation data csv file, the delimiter to use")

    args = parser.parse_args()

    df = pd.read_csv(args.train_data)
    captions = df[args.caption_key].tolist()

    augmented_captions = [augment_caption(caption) for caption in captions]
    df['augment_caption'] = pd.DataFrame({'augment_caption': augmented_captions})

    fname = f'{args.train_data[:-4]}_bow{args.train_data[-4:]}'
    df.to_csv(fname)