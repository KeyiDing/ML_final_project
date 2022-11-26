import re
import pandas as pd
import numpy as np

df = pd.read_csv('labeled_data.csv')

replacements = [('https?://[^ ]+', ""),
                ("@[^\s]*:", ""),
                ("@[^\s]*", ""),
                ("&#[0-9]*;", ""),
                (r'(\w)\1+', r'\1'),
                ('#[^\s]* ', ' '),
                ("&.{,10};", ''),
                ('!*\sRT', ''),
                ('[^A-Za-z0-9 ]+', ''),
                (r'(\s)\1+', r'\1')]


def formatStr(tweet):
    temp = tweet
    for pat, rep in replacements:
        temp = re.sub(pat, rep, temp)

    temp = temp.lower()
    return temp


df.loc[:, "tweet"] = df.tweet.apply(formatStr)
df['length'] = df['tweet'].apply(lambda x: len(x))
df = df[df['length'] > 3]
df.drop(columns=['length'], inplace=True)
df.to_csv('a.csv', index=False)
