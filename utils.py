from textblob import TextBlob
import re
import nltk
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import math
stopwords = ['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now']
dc = ['because', 'since', 'when', 'thus', 'however', 'although', 'for example', 'for instance', 'this']

def clean_puncs(data2):
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for i in punc:
        data2 = data2.replace(i+' ', ' ')
    return data2

def clean_stop(data2):
    stopwords = ['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now']
    stopwordc = [s.capitalize() for s in stopwords]
    for i in stopwords:
        data2 = data2.replace(' '+ i+ ' ', ' ')
    for i in stopwordc:
        data2 = data2.replace(' '+ i+ ' ', ' ')
    return data2

def splitsubs(ww2b):
    ign = ['fig.','table', 'example', '•', 'activity', '=', '+']
    subs = []
    sub = []
    st = ''
    f = 0
    for s in ww2b.sentences:
        s2 = str(s)
        if re.search('^[0-9]\.[0-9].*[A-Z]+.*',s2 ):
            if st:
                sub.append(st)
                subs.append(sub)
            st = [] 
            sub = [s]
            f = 1
        elif f:
            f2 = 0
            for i in clean_puncs(str(s)).split(' '):
                if i.lower() in ign:
                    f2 =1
            if f2 == 0 and len(s2) > 10:
                st.append(s)
    sub.append(st)
    subs.append(sub)
    return subs

def sim(sb, headb):
    c = 0
    for i in sb.tags:
        if i[0].lower() not in stopwords:
            for j in headb.tags:
                if i[0].lower() == j[0].lower():
                    if i[1] in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']:
                        c = c+1
    l = len(clean_puncs(clean_stop(str(sb))).split(' '))
    return c/l
def gen_features(sub):
    headb = sub[0]
    cont = sub[1]
    head = str(headb)
#     print('contb.= ', contb)
    f = 1
    feats = {}
    lmax = 0
    contlen = len(cont)
    k = 1
    for s in cont:
        feat = []

        #feature f
        if f:
            feat.append(1)
            f = 0
        else:
            feat.append(0)

        #feature sim
        feat.append(sim(s, headb))

        #feature abb
        if re.search("\b[A-Z]{2,}\b", str(s)):
            feat.append(1)
        else:
            feat.append(0)

        #feature super
        c = 0
        for i in s.tags:
            if i[1] == 'JJS':
                c = c+1
        feat.append(c/len(str(s)))
        #feature pos
        feat.append(k/contlen)
        k = k+1

        #feature discon
        dc = ['because', 'since', 'when', 'thus', 'however', 'although', 'for example', 'for instance']
        if str(s).split(' ')[0].lower() in dc:
            feat.append(1)
        else:
            feat.append(0)

        #feature l
        l = len(clean_puncs(str(s)).split(' '))
        if l > lmax:
            lmax = l
        feat.append(l)
        #feature n, pn
        n = 0
        pn = 0
        for i in s.tags:
            if i[1] in ['NN', 'NNS', 'NNP', 'NNPS']:
                n += 1
            elif i[1] in ['PRP', 'PRP$']:
                pn += 1
        feat.append(n/l)
        feat.append(pn/l)
        # feat = [f(n), sim(p), abb(p), super(p), pos(extremes), discon(bn),l(mids), n(p), pn(n) ]
        feats[s] = feat
    for i in feats:
        feats[i][6] /= lmax
    return feats

def clean(data2):
    stopwords = ['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now']
    for i in stopwords:
        data2 = data2.replace(' '+ i+ ' ', ' ')
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for i in punc:
        data2 = data2.replace(i, ' ')
    data2 = data2.replace('\n', ' ')
    data2 = data2.replace('\t', ' ')
    data2 = data2.lower()
    return data2
