#from python_speech_features import mfcc as mfc
from python_speech_features import sigproc
import python_speech_features
from audiolazy.lazy_lpc import lpc
import scipy.io.wavfile as wav
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import time
import librosa
import subprocess
import parselmouth

class speech_feat_extract():
    
    def __init__(self, sessions, datapath, categories = ['ang', 'exc', 'neu', 'sad']):
        self.sessions = sessions
        self.datapath = datapath
        self.category_list = categories
        self.id2label = {}
        self.label2ids = {}
        self.dimvar = {}

        def parse_groundtruth(fpath):
            with open(fpath) as fil:
                for line in fil:
                    if line.startswith('['):
                        words = line.rstrip().split('\t')
                        self.id2label[words[1]] = words[2]
                        if self.label2ids.has_key(words[2]):
                            self.label2ids[words[2]].append(words[1])
                        else:
                            self.label2ids[words[2]] = [words[1]]
                        self.dimvar[words[1]] = [float(i) for i in words[3][1:-1].split(', ')] 
        
        for session in sessions:
            for subdir, dirs, files in os.walk(datapath + session + '/dialog/EmoEvaluation/all'):
                for f in files:
                    fpath = subdir + os.sep + f
                    if fpath.endswith('.txt'):
                        parse_groundtruth(fpath)
                        
        self.cat2catid = {}
        for i in range(len(categories)):
            self.cat2catid[categories[i]] = i+1
            
        self.wavpaths = self.filepath_helper('/sentences/wav', '.wav')
        print len(self.wavpaths)

    def pad_sequence_into_array(self, Xs, maxlen=200, truncating='post', padding='post', value=0.):
    	Nsamples = len(Xs)
    	if maxlen is None:
            lengths = [s.shape[0] for s in Xs]    # 'sequences' must be list, 's' must be numpy array, len(s) return the first dimension of s
            maxlen = np.max(lengths)

        Xout = np.ones(shape=[Nsamples, maxlen] + list(Xs[0].shape[1:]), dtype=Xs[0].dtype) * np.asarray(value, dtype=Xs[0].dtype)
        Mask = np.zeros(shape=[Nsamples, maxlen], dtype=Xout.dtype)
        for i in range(Nsamples):
            x = Xs[i]
            if truncating == 'pre':
                trunc = x[-maxlen:]
            elif truncating == 'post':
                trunc = x[:maxlen]
            else:
                raise ValueError("Truncating type '%s' not understood" % truncating)
            if padding == 'post':
                Xout[i, :len(trunc)] = trunc
                Mask[i, :len(trunc)] = 1
            elif padding == 'pre':
                Xout[i, -len(trunc):] = trunc
                Mask[i, -len(trunc):] = 1
            else:
                raise ValueError("Padding type '%s' not understood" % padding)
        return Xout, Mask
            
    def filepath_helper(self, relpath, filetype):
        filepaths = []
        for session in self.sessions:
            for subdir, dirs, files in os.walk(self.datapath + session + relpath):
                for f in files:
                    if f.endswith(filetype) and self.id2label[f[:-4]] in self.category_list:
                        filepaths.append([subdir + os.sep + f, f[:-4]])
        return filepaths                        
        
    def compute_mfcc(self, pad, save=False):
        filepaths = self.wavpaths
        self.mfcc = {}
        k = 0
        for f in filepaths:
            if k%1000 == 0:
                print k
            k += 1
            sound = parselmouth.Sound(f[0])
            melspectrogram = parselmouth.praat.call(sound, "To MelSpectrogram", 0.025, 0.01, 100.0, 100.0, 0.0)
            mfcc = parselmouth.praat.call(melspectrogram, "To MFCC", 13)
            #mfcc = sound.to_mfcc(number_of_coefficients=13, window_length=0.025, time_step=0.01)
            mfcc_mat = parselmouth.praat.call(mfcc, "To Matrix").as_array()
            #self.mfcc[f[1]] = mfcc.to_matrix_features(include_energy=True).as_array()
            if pad > 0:
                mfcc_mat, _ = self.pad_sequence_into_array(mfcc_mat, maxlen=pad)
            self.mfcc[f[1]] = mfcc_mat.transpose()
        if save:
            with open('../feat/mfcc.pkl', 'wb') as f:
                pickle.dump(self.mfcc, f)
        
    def compute_zcr(self, pad, save=False):
        filepaths = self.wavpaths
        for f in filepaths:
            sig, sr = librosa.load(f[0])
            zcr_rate = librosa.feature.zero_crossing_rate(sig)[0]
            zcr_rate = np.reshape(zcr_rate, (-1,zcr_rate.shape[0]))
            A = librosa.feature.delta(zcr_rate, order = 1)
            print A.shape
            m = np.concatenate((zcr_rate, A), axis = 0)
            A = librosa.feature.delta(zcr_rate, order = 2)
            self.zcr[f[1]], _ = self.pad_sequence_into_array(np.concatenate((m, A), axis = 0), maxlen=pad)
        if save:
            with open('../feat/zcr.pkl', 'wb') as f:
                pickle.dump(self.zcr, f)
        
    def compute_pitch(self, pad, save=False):
        filepaths = self.wavpaths
        for f in filepaths:
            subprocess.call(['/usr/bin/praat', '--run', 'extract_pitch.praat', f[0]])
            pitch = []
            with open('temp.pitch') as fil:
                for i in fil:
                    val = i.split()
                    if val[1] != '--undefined--' and val[0] != 'Pitch':
                        pitch.append(float(val[1]))
            temp = np.reshape(np.array(pitch), (1, -1))
            print temp.shape
            self.pitch[f[1]] = self.pad_sequence_into_array(temp, maxlen=pad)
        if save:
            with open('feat/pitch.pkl', 'wb') as f:
                pickle.dump(self.pitch, f)

    def compute_chroma_cens(self, pad, save=False):
        filepaths = self.wavpaths
        self.chroma_cens = {}
        for f in filepaths:
            sig, sr = librosa.load(f[0])
            self.chroma_cens[f[1]] = self.pad_sequence_into_array(librosa.feature.chroma_cens(sig, n_chroma=12), maxlen=pad)
        if save:
            with open('feat/chroma_cens.pkl', 'wb') as f:
                pickle.dump(self.chroma_cens, f)

    def compute_voiceQuality(self, save=False):
        filepaths = self.wavpaths
        self.voiceQuality = {}
        j = 0
        for f in filepaths:
            subprocess.call(['/usr/bin/praat', '--run', 'extract_voiceQuality.praat', f[0]])
            with open('temp.voiceQuality') as fil:
                val = []
                for i in fil:
                    val += [float(k) if k!='--undefined--' else 0 for k in i.rstrip().split(' ')]
                if j%100 == 0:
                    print j
                j += 1
            self.voiceQuality[f[1]] = val
        if save:
            with open('../feat/voiceQuality.pkl', 'wb') as f:
                pickle.dump(self.voiceQuality, f)

    def compute_lpcc(self, pad, save=False):
        filepaths = self.wavpaths
        self.lpcc = {}
        k = 0
        for f in filepaths:
            if k % 1000 == 0:
                print k
            k += 1
            sound = parselmouth.Sound(f[0])
            lpc = parselmouth.praat.call(sound, "To LPC (autocorrelation)", 12, 0.025, 0.01, 50.0)
            lpc_mat = parselmouth.praat.call(lpc, "Down to Matrix (lpc)")
            if pad > 0:
                lpc_mat, _ = self.pad_sequence_into_array(lpc_mat.values.transpose(), maxlen=pad)
            self.lpcc[f[1]] = lpc_mat.transpose()
        if save:
            with open('../feat/lpcc.pkl', 'wb') as f:
                pickle.dump(self.lpcc, f)
            
    def compute_rmfcc(self, pad, save=False):
        filepaths = self.wavpaths
        self.rmfcc = {}
        k = 0
        for f in filepaths:
            if k%1000 == 0:
                print k
            k += 1
            sound = parselmouth.Sound(f[0])
            lpc = parselmouth.praat.call(sound, "To LPC (autocorrelation)", 12, 0.025, 0.01, 50.0)
            residual = parselmouth.praat.call([sound, lpc], "Filter (inverse)")
            melspectrogram = parselmouth.praat.call(residual, "To MelSpectrogram", 0.025, 0.01, 100.0, 100.0, 0.0)
            rmfcc = parselmouth.praat.call(melspectrogram, "To MFCC", 13)
            #mfcc = sound.to_mfcc(number_of_coefficients=13, window_length=0.025, time_step=0.01)
            rmfcc_mat = parselmouth.praat.call(rmfcc, "To Matrix").as_array()
            #self.mfcc[f[1]] = mfcc.to_matrix_features(include_energy=True).as_array()
            if pad > 0:
                rmfcc_mat, _ = self.pad_sequence_into_array(rmfcc_mat, maxlen=pad)
            self.rmfcc[f[1]] = rmfcc_mat.transpose()
        if save:
            with open('../feat/rmfcc.pkl', 'wb') as f:
                pickle.dump(self.rmfcc, f)

    def compute_stat(self, feat, save=False):
        if feat == 'mfcc':
            X = self.mfcc
        elif feat == 'zcr':
            X = self.zcr
        elif feat == 'lpcc':
            X = self.lpcc
        elif feat == 'rmfcc':
            X = self.rmfcc
        
        X_mean = np.empty((0,X.shape[1]), float)
        X_var = np.empty((0,X.shape[1]), float)
        X_min = np.empty((0,X.shape[1]), float)
        X_max = np.empty((0,X.shape[1]), float)
        label = []
        fnames = []
        for k,v in X.items():
            A = np.mean(v, axis=0)
            X_mean = np.concatenate((X_mean, np.reshape(A, (-1, A.shape[0]))), axis=0)
            A = np.var(v, axis=0)
            X_var = np.concatenate((X_var, np.reshape(A, (-1, A.shape[0]))), axis=0)
            A = np.min(v, axis=0)
            X_min = np.concatenate((X_min, np.reshape(A, (-1, A.shape[0]))), axis=0)
            A = np.max(v, axis=0)
            X_max = np.concatenate((X_max, np.reshape(A, (-1, A.shape[0]))), axis=0)
            label.append(self.cat2catid[self.id2label[k]])
            fnames.append(k)

        feat_stat = np.concatenate((X_mean, X_var, X_min, X_max), axis=1)
        self.label = label
        self.fnames = fnames
        if feat == 'mfcc':
            self.mfcc_stat = feat_stat
        elif feat == 'lpcc':
            self.lpcc_stat = feat_stat            
        elif feat == 'zcr':
            self.zcr_stat = zcr_stat
        elif feat == 'rmfcc':
            self.lpcc_stat = lpcc_stat

        if save:
            with open('../feat/' + feat + '_stat.pkl', 'wb') as f:
                pickle.dump(feat_stat, f)
