from textblob import TextBlob
import re
import nltk
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import math
import utils
from sklearn.externals import joblib
stopwords = ['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now']
dc = ['because', 'since', 'when', 'thus', 'however', 'although', 'for example', 'for instance', 'this']
file = open('inputb.txt', 'r')
data = file.read()
data = data.replace('\n[.]*', '. ')
data = data.replace('\n', '. ')
data = data.replace('\t', ' ')
data = re.sub(' +', ' ', data)
datab = TextBlob(data)

subs = utils.splitsubs(datab)
with open('biodata.json') as json_file:
	test_data = json.load(json_file)
	sc1 = [d["text"] for d in test_data]

superdf = pd.DataFrame([], columns = ['sentence', 'f','sim', 'abb','super', 'pos', 'discon', 'l', 'n', 'pn', 'score'])
for sub in subs:
	feats  = utils.gen_features(sub)
	featsl = []
	for k in feats:
		l = []
		l.append(str(k))
		for i in feats[k]:
			l.append(float(i))
		score = 0
		if k in sc1:
			score = 1
		l.append(score)
		featsl.append(l)
		
	df = pd.DataFrame(featsl, columns = ['sentence', 'f','sim', 'abb','super', 'pos', 'discon', 'l', 'n', 'pn', 'score'])
	superdf = pd.concat([superdf, df])

y = superdf.score
X = superdf[['f','sim', 'abb','super', 'pos', 'discon', 'l', 'n', 'pn']]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
BioModel = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
BioModel.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(val_X, val_y)], verbose=False)

joblib.dump(BioModel, 'bio.joblib.dat')

####################################################################

file = open('inputp.txt', 'r')
data = file.read()
data = data.replace('\n[.]*', '. ')
data = data.replace('\n', '. ')
data = data.replace('\t', ' ')
data = re.sub(' +', ' ', data)
datab = TextBlob(data)

subs = utils.splitsubs(datab)
with open('phydata.json') as json_file:
	test_data = json.load(json_file)
	sc1 = [d["text"] for d in test_data]

superdf = pd.DataFrame([], columns = ['sentence', 'f','sim', 'abb','super', 'pos', 'discon', 'l', 'n', 'pn', 'score'])
for sub in subs:
	feats  = utils.gen_features(sub)
	featsl = []
	for k in feats:
		l = []
		l.append(str(k))
		for i in feats[k]:
			l.append(float(i))
		score = 0
		if k in sc1:
			score = 1
		l.append(score)
		featsl.append(l)
		
	df = pd.DataFrame(featsl, columns = ['sentence', 'f','sim', 'abb','super', 'pos', 'discon', 'l', 'n', 'pn', 'score'])
	superdf = pd.concat([superdf, df])

y = superdf.score
X = superdf[['f','sim', 'abb','super', 'pos', 'discon', 'l', 'n', 'pn']]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
PhyModel = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
PhyModel.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(val_X, val_y)], verbose=False)

joblib.dump(PhyModel, 'phy.joblib.dat')

####################################################################

file = open('inputc.txt', 'r')
data = file.read()
data = data.replace('\n[.]*', '. ')
data = data.replace('\n', '. ')
data = data.replace('\t', ' ')
data = re.sub(' +', ' ', data)
datab = TextBlob(data)

subs = utils.splitsubs(datab)
with open('chemdata.json') as json_file:
	test_data = json.load(json_file)
	sc1 = [d["text"] for d in test_data]

superdf = pd.DataFrame([], columns = ['sentence', 'f','sim', 'abb','super', 'pos', 'discon', 'l', 'n', 'pn', 'score'])
for sub in subs:
	feats  = utils.gen_features(sub)
	featsl = []
	for k in feats:
		l = []
		l.append(str(k))
		for i in feats[k]:
			l.append(float(i))
		score = 0
		if k in sc1:
			score = 1
		l.append(score)
		featsl.append(l)
		
	df = pd.DataFrame(featsl, columns = ['sentence', 'f','sim', 'abb','super', 'pos', 'discon', 'l', 'n', 'pn', 'score'])
	superdf = pd.concat([superdf, df])

y = superdf.score
X = superdf[['f','sim', 'abb','super', 'pos', 'discon', 'l', 'n', 'pn']]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
ChemModel = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
ChemModel.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(val_X, val_y)], verbose=False)

joblib.dump(PhyModel, 'chem.joblib.dat')
