from textblob import TextBlob
import re
import nltk
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import math
import unidecode
import sys
import utils
from sklearn.externals import joblib

stopwords = ['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now']
dc = ['because', 'since', 'when', 'thus', 'however', 'although', 'for example', 'for instance', 'this']

file = open(sys.argv[1], 'r')
data = file.read()
data = data.replace('\n', '. ')
data = data.replace('\t', ' ')
data = re.sub(' +', ' ', data)
datab = TextBlob(data)

subs = utils.splitsubs(datab)
# print(subs)

superdf = pd.DataFrame([], columns = ['sentence', 'f','sim', 'abb','super', 'pos', 'discon', 'l', 'n', 'pn', 'score'])
for sub in subs:
	feats  = utils.gen_features(sub)
	featsl = []
	for k in feats:
		l = []
		l.append(str(k))
		for i in feats[k]:
			l.append(float(i))
		featsl.append(l)
		# print(featsl)

	df = pd.DataFrame(featsl, columns = ['sentence', 'f','sim', 'abb','super', 'pos', 'discon', 'l', 'n', 'pn'])
	superdf = pd.concat([superdf, df], sort = False)
X = superdf[['f','sim', 'abb','super', 'pos', 'discon', 'l', 'n', 'pn']]
predictions = None
if sys.argv[2] == 'b':
		model = joblib.load('bio.joblib.dat')
		predictions = model.predict(X)
		a = 0.4
elif sys.argv[2] == 'c':
		model = joblib.load('chem.joblib.dat')
		predictions = model.predict(X)
		a = 0.13
elif sys.argv[2] == 'p':
		model = joblib.load('phy.joblib.dat')
		predictions = model.predict(X)
		a = 0.14

stop2 = ['as', 'i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves',
'this','that']
m = max(predictions)
impsen = []
predictions = predictions/m
for i in range(len(predictions)):
    if predictions[i] > a:
        s = superdf.iloc[i][0]
        # print('YEET',s)
        if not (re.search('[?!]$', s) or s.split(' ')[0].lower() in (dc + stop2) or 'fig.' in s.lower()):
            s = re.sub('\.[ ]*[.]+', '.', s)
            impsen.append(unidecode.unidecode(str(s)))

data2 = utils.clean(data)

l = len(data2.split(' '))
tf= {}
for i in data2.split(' '):
	i = i.strip()
	if i in tf:
		tf[i] = tf[i] + 1/l
	else:
		tf[i] = 1/l

ls = len(datab.sentences)
idf= {}
for i in data2.split(' '):
	i = i.strip()
	if i in idf:
		idf[i] = idf[i] + 1
	else:
		idf[i] = 1
for i in idf:
	idf[i] = math.log10(ls/idf[i])
score = {}
for s in tf:
	score[s] = tf[s] * idf[s]
smax = score[max(score)]
print(score)
if sys.argv[2] == 'b':
	model = joblib.load('bio.joblib.dat')
	predictions = model.predict(X)
	for s in score:
		score[s] /= smax
	b = 0.3
	c = 0.8
elif sys.argv[2] == 'c':
	b = smax/3
	c = smax/1.8
elif sys.argv[2] == 'p':
	b = smax/3
	c = smax/1.5
output = []
for s in impsen:
	outs = {}
	wl = set([])
	sb = TextBlob(s)
	for w in sb.tags:
		if w[1] in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'CD']:
			if w[0].lower() in score:
				if score[w[0].lower()] > b and score[w[0].lower()] < c:
					wl.add(w[0].lower())
	outs['text'] = s
	outs['fibs'] = wl
	if wl:
		output.append(outs)
# print(output)
outf = open('output.json', 'w+')
for i in output:
	outf.write(str(i) + '\n')
outf.close()
