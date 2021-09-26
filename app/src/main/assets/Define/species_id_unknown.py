# species_id_unknown.py 
# predict bird species from songs and calls.
# Gerard Gilliland MODEL Software, Inc. 
# https://www.modelsw.com/Android/BirdingViaMic/BirdingViaMic.php
# gerardg@modelsw.com

# based on speaker_id.py
# performs a speaker_id experiments with SincNet.
# Mirco Ravanelli 
# Mila - University of Montreal 
# https://github.com/mravanelli/SincNet
# July 2018

# populated using bird songs from https://www.xeno-canto.org

# build database in SQLite from IOC World Bird List by Frank Gill.
# https://www.worldbirdnames.org/new/  

# using Convolutional Neural Network model built on Ubuntu 20, Python 3.6, Nvidia Quadro GV100 GPU
# converted to use only CPU.
# run using terminal$ python3 species_id_prediction.py

# run in Android Studio terminal$ from qpython3-3x-v3.0.0 in AndroidStudio using 
# https://github.com/qpython-android/qpython3/releases

# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# soundfile ( conda install -c conda-forge pysoundfile) 

# this is species_id_prediction.py modified to read an original bird song
# it uses a random specie that exists in /home/gerardg/storage/Named1133/
# regardless of whether it is in the model either in the trained or test area 
# it could most likely be missing -- there are 1133 species but the current model only has 219 trained
# the concept is to be able to use it in Birding Via Mic -- but I am not there yet.
# the Sxxxx name is convertered to a Bird name and looked up in Named1133
# it is compressed to 30 db and sent to the model.

import os
import librosa  # https://github.com/Subtitle-Synchronizer/jlibrosa#readme
#import soundfile as sf
#https://pythonrepo.com/repo/pytorch-audio-python-deep-learning
#import torchaudio as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import sys
import numpy as np
import random
from dnn_models import MLP,flip
from dnn_models import SincNet as CNN 
from data_io import ReadList,read_conf,str_to_bool
from random import randrange
import json

import time
import datetime
start_time = datetime.datetime.now()

import sqlite3
db_file = 'BirdSongs111.db'
link = sqlite3.connect(db_file)
HOP_LENGTH = 512
wshift=220 # int(fs*cw_shift/1000.00) --> 22050 * 20 / 1000 -- keeps phrases separate by 220 cycles (0.01 of a second)


def calc_cntr(mode, cnt):
    zerocount = 0
    zero = " "
    #totcount = np.zeros([class_size])
    maxcount = 0    
    bestclass = -1
     
    for t in range(N_fr):
        thisClass = cnt[t]
        totcount[thisClass] += 1

    for t in range(class_size):
        if totcount[t] > 5:
            sp = species[t]
            rf = sp[1:]
            com = ConvertToName(rf)
            print ("class: ", t, " count: ", totcount[t], com)
            #print ("class: ", t, " count: ", totcount[t])
        if maxcount < totcount[t]:
            maxcount = totcount[t]
            bestclass = t

    specie = species[bestclass]
    #print ("specie: ", specie)
    ref = specie[1:]
    #print ("ref: ", ref)
    comnam = ConvertToName(ref)     
    print ("calc_cntr: ", bestclass, " maxcount: ", maxcount, " ref: ", ref, comnam)
    return bestclass, maxcount
    

# Reading cfg file
options=read_conf()

#[data] -- THIS POINTS TO A DIFFERENT DATASET THAN CFG FILE
db = 20 # only change this once per run. I might decide 32db is better but then all should be 32db
data_folder="./Song/" 

#[windowing]
fs=22050
cw_len=200
cw_shift=10

#[cnn]
cnn_N_filt=list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt=list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len=list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp=str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp=str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm=list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm=list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act=list(map(str, options.cnn_act.split(',')))
cnn_drop=list(map(float, options.cnn_drop.split(',')))


#[dnn]
fc_lay=list(map(int, options.fc_lay.split(',')))
fc_drop=list(map(float, options.fc_drop.split(',')))
fc_use_laynorm_inp=str_to_bool(options.fc_use_laynorm_inp)
fc_use_batchnorm_inp=str_to_bool(options.fc_use_batchnorm_inp)
fc_use_batchnorm=list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
fc_use_laynorm=list(map(str_to_bool, options.fc_use_laynorm.split(',')))
fc_act=list(map(str, options.fc_act.split(',')))

#[class]
class_lay=list(map(int, options.class_lay.split(',')))
class_drop=list(map(float, options.class_drop.split(',')))
class_use_laynorm_inp=str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp=str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm=list(map(str_to_bool, options.class_use_batchnorm.split(',')))
class_use_laynorm=list(map(str_to_bool, options.class_use_laynorm.split(',')))
class_act=list(map(str, options.class_act.split(',')))


seed=int(options.seed)


# setting seed
torch.manual_seed(seed)
np.random.seed(seed)

# loss function
cost = nn.NLLLoss()

  
# Converting context and shift in samples
wlen=int(fs*cw_len/1000.00)
wshift=int(fs*cw_shift/1000.00)

# Batch_dev
Batch_dev=128


# Feature extractor CNN
CNN_arch = {'input_dim': wlen,
          'fs': fs,
          'cnn_N_filt': cnn_N_filt,
          'cnn_len_filt': cnn_len_filt,
          'cnn_max_pool_len':cnn_max_pool_len,
          'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
          'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
          'cnn_use_laynorm':cnn_use_laynorm,
          'cnn_use_batchnorm':cnn_use_batchnorm,
          'cnn_act': cnn_act,
          'cnn_drop':cnn_drop,          
          }

CNN_net=CNN(CNN_arch)
CNN_net.cpu()

# Loading label dictionary
lab_dict=np.load("species_lab.npy",allow_pickle=True).item()
#print (lab_dict)

DNN1_arch = {'input_dim': CNN_net.out_dim,
          'fc_lay': fc_lay,
          'fc_drop': fc_drop, 
          'fc_use_batchnorm': fc_use_batchnorm,
          'fc_use_laynorm': fc_use_laynorm,
          'fc_use_laynorm_inp': fc_use_laynorm_inp,
          'fc_use_batchnorm_inp':fc_use_batchnorm_inp,
          'fc_act': fc_act,
          }

DNN1_net=MLP(DNN1_arch)
DNN1_net.cpu()


DNN2_arch = {'input_dim':fc_lay[-1], 
          'fc_lay': class_lay, 
          'fc_drop': class_drop, 
          'fc_use_batchnorm': class_use_batchnorm,
          'fc_use_laynorm': class_use_laynorm,
          'fc_use_laynorm_inp': class_use_laynorm_inp,
          'fc_use_batchnorm_inp':class_use_batchnorm_inp,
          'fc_act': class_act,
          }


DNN2_net=MLP(DNN2_arch)
DNN2_net.cpu()

def ConvertToName(ref):
    #print ("ref: ", ref)
    rs = link.cursor()
    qry = "Select CommonName from CodeName where Ref = " + str(ref)
    #print ("qry: ", qry)
    rs.execute(qry)
    comnam = rs.fetchall()
    #comnam = rs.fetchone()
    comnam = str(comnam)
    inx = comnam.find(",")
    comnam = comnam[:inx]
    inx = comnam.find("(")+1
    comnam = comnam[inx:]
    comnam = comnam[1:-1]
    return comnam

def get_species():
    species_list = "./species_list.txt"
    with open(species_list, 'r') as f:
        species = json.loads(f.read())
    return species

species = get_species()

def norm(data):
    return librosa.util.normalize(data, np.inf, axis=0, threshold=None, fill=None)

# if model exists and is saved
pt_file = "./model_raw.pkl"
if os.path.isfile(pt_file): 
    checkpoint = torch.load(pt_file, map_location='cpu')
    CNN_net.load_state_dict(checkpoint['CNN_model_par'])
    DNN1_net.load_state_dict(checkpoint['DNN1_model_par'])
    DNN2_net.load_state_dict(checkpoint['DNN2_model_par'])
else:
    print ("Model: " + pt_file + " is missing")
    exit()

# Validation
      
CNN_net.eval()
DNN1_net.eval()
DNN2_net.eval()
test_flag=1 
loss_sum=0
acc_sum=0
acc_sum_snt=0
class_size = int(class_lay[0]) # 505
totcount = [0] * class_size

torch.no_grad()  # don't mess with the model just use it to predict the bird
# note: my files are wav or m4a - 16 bit - big endian - mono - 22050

print ("data_folder: ", data_folder)
wav_name = random.choice(os.listdir(data_folder))
print ("wav_name: ", wav_name)
inx = wav_name.find("_")
random_specie = wav_name[:inx]
print ("random_specie: ",  random_specie)
wav_name = data_folder + wav_name

lab_batch = 0
print ("wav_name: ", wav_name)
audio, sr = librosa.load(wav_name)
audio = norm(audio)
newlen = len(audio)
newwav_test = np.zeros(newlen) # more than enough 
this_len = 0
# intervals is a list of start and end locations containing high db in the file that you want to keep
intervals = librosa.effects.split(audio, top_db=db, ref=np.max, frame_length=1024, hop_length=HOP_LENGTH)       
for i,j in intervals:
    #print ("intervals i:", i, " j:", j, " diff: ", (j-i), " this_len: ", this_len)  
    newwav_test[this_len:(this_len+j-i)] = audio[i:j]    
    this_len += (j-i)+wshift # keeping phrases separate by 220 cycles
    
newwav_test = newwav_test[:this_len]

# split signals into chunks
beg_samp=0
end_samp=wlen

N_fr=int((this_len-wlen)/(wshift))+1
print ("this_len:", this_len, " number frames:", N_fr) 

sig_arr=torch.zeros([Batch_dev,wlen]).float().cpu().contiguous()
lab= Variable((torch.zeros(N_fr)+lab_batch).cpu().contiguous().long())
pout=Variable(torch.zeros(N_fr,class_lay[-1]).float().cpu().contiguous())
count_fr=0
count_fr_tot=0
while end_samp<this_len:
    sig_arr[count_fr:]=torch.FloatTensor(newwav_test[beg_samp:end_samp])
    beg_samp=beg_samp+wshift
    end_samp=beg_samp+wlen
    count_fr=count_fr+1
    count_fr_tot=count_fr_tot+1
    if count_fr==Batch_dev:
        print (count_fr, " frames loaded into model inp")
        inp=Variable(sig_arr)
        pout[count_fr_tot-Batch_dev:count_fr_tot,:]=DNN2_net(DNN1_net(CNN_net(inp)))
        count_fr=0
        sig_arr=torch.zeros([Batch_dev,wlen]).float().cpu().contiguous()
            
if count_fr>0:
    print (count_fr, " frames loaded into model inp")        
    inp=Variable(sig_arr[0:count_fr])
    pout[count_fr_tot-count_fr:count_fr_tot,:]=DNN2_net(DNN1_net(CNN_net(inp)))
  
pred=torch.max(pout,dim=1)[1]
loss = cost(pout, lab.long())
acc = torch.mean((pred==lab.long()).float())
#print ("pred: ", pred, " loss:", loss)
    
t1 = lab.flatten().clone().detach()
t2 = torch.argmax(pout, 1).clone().detach()
#print ("t1 = y.flatten(): ", t1, "\n pout:", pout, "\n t2 = torch.argmax(pout, 1):", t2 )
#teq = torch.eq(t1, t2).clone().detach()
#print("match: torch.eq(t1, t2))", teq)

result = t2.cpu()
bestclass, maxcount = calc_cntr('valid', result) 

loss_sum=loss_sum+loss.detach()
acc_sum=acc_sum+acc.detach()
            
acc_tot_dev_snt=acc_sum_snt/N_fr*100
loss_tot_dev=loss_sum/N_fr
acc_tot_dev=acc_sum/N_fr*100

specie = species[bestclass]
ref = specie[1:]
comnam = ConvertToName(ref) 
#enclosed_random_specie = "'" + random_specie + "'"
#if comnam == enclosed_random_specie:
print ("random_specie: ", random_specie)
if comnam == random_specie:
    prediction = "successful."
else:
    prediction = "fails."
print ("predict: " , bestclass,  " ref: ", ref, " prediction: ", comnam, prediction)
x = maxcount/N_fr * 100
print("Percent accuracy " + str(maxcount) + "/" + str(N_fr) +" = {:.2f}".format(x) + " %");  
                        
end_time = datetime.datetime.now()
print("Prediction Finished: ", end_time)
difference = end_time - start_time
print ("prediction time: ", difference)

exit()


