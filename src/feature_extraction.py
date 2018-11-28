from features import speech_feat_extract
from sklearn.preprocessing import label_binarize
from model import LSTM_AE, LSTM_dim, LSTM_cat
import numpy as np
import pickle

class LowLevelFeat():
	def __init__(self):
		sess = ['Session'+str(i+1) for i in range(5)]
		categories = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
		self.data = speech_feat_extract(sess, '../../IEMOCAP_full_release/', categories)

	def feat_extract(self, arg, pad=100):
		if arg == 'mfcc':
			self.data.compute_mfcc(pad)
			feat = self.data.mfcc
		elif arg == 'zcr':
			self.data.compute_zcr(pad)
			feat = self.data.zcr
		elif arg == 'pitch':
			self.data.compute_pitch(pad)
			feat = self.data.pitch
		elif arg == 'lpcc':
			self.data.compute_lpcc(pad)
			feat = self.data.lpcc
		elif arg == 'chroma':
			self.data.compute_chroma_cens(pad)
			feat = self.data.chroma_cens
		elif arg == 'VQ':
			self.data.compute_voiceQuality()
			feat = self.data.voiceQuality
		elif arg == 'rmfcc':
			self.data.compute_rmfcc(pad)
			feat = self.data.rmfcc
		X = []
		y = []
		subject = []
		id_ = []
		dim = []
		for k,v in feat.items():
			X.append(v)
			y.append(self.data.cat2catid[self.data.id2label[k]])
			subject.append(k.split('_')[0])
			id_.append(k)
			dim.append(np.array(self.data.dimvar[k]))
		temp = {'feat':X, 'cat':y, 'subject':subject, 'id':id_, 'dim':dim}
		with open('../feat/' + arg + '_' + str(pad) + '.pkl', 'wb') as f:
			pickle.dump(temp, f)


class HighLevelFeat():
	def __init__(self, fi):
		with open('feat/' + fi[0] + '_' + fi[1] + '.pkl') as f:
			self.data = pickle.load(f)

	def extract_feat(self, model_type):
		X = self.data['feat']
		y = self.data['cat']
		if model_type == 'AE':
			model = LSTM_AE(X.shape[1], fi[1], X.shape[2])
		print model.train(X, epochs=20)
		feat_X = model.feature(X)
		with open(model_type + '_feat_' + fi[0] + '_' + str(fi[1]) + '.pkl', 'wb') as f:
			pickle.dump({'feat':feat_X, 'cat':y, 'subject':self.data['subject'], 'id':self.data['id_']}, f)

#llf = LowLevelFeat()
#llf.feat_extract('mfcc', 200)
#llf.feat_extract('lpcc', 200)
#llf.feat_extract('rmfcc', 200)
#llf.feat_extract('VQ')
#llf.feat_extract('mfcc', 150)
#llf.feat_extract('lpcc', 150)
#llf.feat_extract('rmfcc', 150)
#llf.feat_extract('mfcc', 250)
#llf.feat_extract('lpcc', 250)
#llf.feat_extract('rmfcc', 250)
#llf.feat_extract('zcr', 100)
#llf.feat_extract('zcr', 50)
#llf.feat_extract('zcr', 75)
