# speaker_id.py
# Mirco Ravanelli 
# Mila - University of Montreal 

# July 2018

# Description: 
# This code performs species_id modeling with SincNet.
 
# How to run it:
"""

cd storage/Bird_SincNetBalanced
check species.cfg
python3 species_id.py

"""

import os 
import soundfile as sf
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.autograd import Variable 

import sys
import numpy as np 
from dnn_models import MLP,flip
from dnn_models import SincNet as CNN 
from data_io import ReadList,read_conf,str_to_bool
import pickle
import json

import sqlite3
db_file = 'BirdSongs111.db'
link = sqlite3.connect(db_file)
import datetime
import time

# check if GPU is available
train_on_gpu = torch.cuda.is_available() # support for CUDA tensor types (on the GPU vs CPU)
if not train_on_gpu: # returns device=none
    print('Bummer!  Training on CPU ...')
    print('try a reboot -- have you done an upgrade ?')
    exit()
else: # returns stream for current device
    print('You are good to go!  Training on GPU ...')

md = open ("MissingDefinition.txt","w")

def create_batches_rnd(batch_size,data_folder,wav_lst,N_snt,wlen,lab_dict,fact_amp):
    
    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    sig_batch=np.zeros([batch_size,wlen])  # two dimensional array batch_size=131 x wlen=int(fs=22050*200/1000.00)=4410
    lab_batch=np.zeros(batch_size)
  
    snt_id_arr=np.random.randint(N_snt, size=batch_size) # N_snt = number of sentences in training
  
    rand_amp_arr = np.random.uniform(1.0-fact_amp,1+fact_amp,batch_size)

    for i in range(batch_size):
        #print ("in create_batches_rnd for i: " + str(i) + " data_folder: " + data_folder + " wav_lst: " + wav_lst[i])
        [signal, fs] = sf.read(data_folder+wav_lst[snt_id_arr[i]]) 

        # accesing to a random chunk
        snt_len=signal.shape[0]
        #snt_beg=np.r0, snt_len-2*wlen-1) # between 0 and 71,491-6401 (why 2* in comments?)andom.randint(snt_len-wlen-1) #randint(
        #snt_beg=np.random.randint(0,snt_len) 
        snt_beg=np.random.randint(snt_len-wlen-1) #randint(0, snt_len-2*wlen-1)
        snt_end=snt_beg+wlen

        channels = len(signal.shape)
        if channels == 2:
            print('WARNING: stereo to mono: '+data_folder+wav_lst[snt_id_arr[i]])
            signal = signal[:,0]
  
        sig_batch[i,:]=signal[snt_beg:snt_end]*rand_amp_arr[i] # populate the zero array with one segment modified by +/-8.3% 
        #lab_batch[i] = lab_dict[wav_lst[i]]
        lab_batch[i]=lab_dict[wav_lst[snt_id_arr[i]]] # read the wave name find it in the dictionary get the label for this signal 

    inp=Variable(torch.from_numpy(sig_batch).float().cuda().contiguous())
    lab=Variable(torch.from_numpy(lab_batch).float().cuda().contiguous())
  
    return inp,lab  



# Reading cfg file
options=read_conf()

#[data]
tr_lst=options.tr_lst # dataset/species/train_list.txt
te_lst=options.te_lst # dataset/species/test_list.txt
pt_file=options.pt_file
class_dict_file=options.lab_dict # 
data_folder=options.data_folder # dataset/species/
output_folder=options.output_folder # output

#[windowing]
fs=int(options.fs)
cw_len=int(options.cw_len)
cw_shift=int(options.cw_shift)

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


#[optimization]
lr=float(options.lr)
batch_size=int(options.batch_size)
N_epochs=int(options.N_epochs) 
N_batches=int(options.N_batches)
N_eval_epoch=int(options.N_eval_epoch)
seed=int(options.seed)

# training list
wav_lst_tr=ReadList(tr_lst)
snt_tr=len(wav_lst_tr) # number of sentences in the train_list.txt = 786 

# test list
wav_lst_te=ReadList(te_lst)
snt_te=len(wav_lst_te) # number of sentences in test_list.txt = 281


# Folder creation
try:
    os.stat(output_folder)
except:
    os.mkdir(output_folder) 
    
    
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
CNN_arch = {'input_dim': wlen,  # this is where wlen becomes input_dim
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
CNN_net.cuda()

# Loading label dictionary
# print ("class_dict_file: ", class_dict_file)
# SincNet requires a prebuilt label dictionary 
# Genre builds it on the fly in data_manager 
lab_dict=np.load(class_dict_file, allow_pickle=True).item()


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
DNN1_net.cuda()


DNN2_arch = {'input_dim':fc_lay[-1] ,
          'fc_lay': class_lay,
          'fc_drop': class_drop, 
          'fc_use_batchnorm': class_use_batchnorm,
          'fc_use_laynorm': class_use_laynorm,
          'fc_use_laynorm_inp': class_use_laynorm_inp,
          'fc_use_batchnorm_inp':class_use_batchnorm_inp,
          'fc_act': class_act,
          }


DNN2_net=MLP(DNN2_arch)
DNN2_net.cuda()

#if pt_file!='none':
if os.path.isfile(output_folder+'/model_raw.pkl'): 
    checkpoint_load = torch.load(output_folder+'/model_raw.pkl') 
    CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
    DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par']) 
    DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par']) 


optimizer_CNN = optim.RMSprop(CNN_net.parameters(), lr=lr,alpha=0.95, eps=1e-8) 
optimizer_DNN1 = optim.RMSprop(DNN1_net.parameters(), lr=lr,alpha=0.95, eps=1e-8) 
optimizer_DNN2 = optim.RMSprop(DNN2_net.parameters(), lr=lr,alpha=0.95, eps=1e-8) 

class_size = int(class_lay[0]) 

trncnt = [0] * class_size   # train number of speaker
vldcnt = [0] * class_size   # valid number of speakers 

def calc_cntr(mode, cnt):
    md.write(mode + "\n") 
    tic = chr(39)
    zerocount = 0
    zero = " "
    totcount = 0
    maxcount = 0     
    for t in range(class_size):
        thiscnt = cnt[t]
        totcount += thiscnt
        if thiscnt == 0:
            comnam = ConvertToName(t)
            zerocount += 1
            #inx = comnam.find(tic)
            #if inx > 0:
            #    inxone = inx + 1
            #    comnam = comnam[:inx] + comnam[inxone:]
            zero += comnam + ":" + str(t) + ", "            
            md.write(comnam + ":" + str(t) + "\n") 
        if maxcount < thiscnt:
            maxcount = thiscnt
    avecount = totcount / class_size
    maxratio = int((maxcount / avecount) + 0.5)
    avecount = int(avecount + 0.5)
    tx = mode + "  totcount: " + str(totcount) + "  failed: " + str(zerocount) + "  at ( " + zero + " )  aveCount: " + str(avecount) + "  maxratio: " + str(maxratio)
    print(tx)
    with open(output_folder+"/res.res", "a") as res_file:
        res_file.write(tx + "\n")   


def ConvertToName(class_num):  # 3
    bc = int(class_num)
    specie = species[bc] # 27000
    ref = specie[1:]
    rs = link.cursor()
    qry = "Select CommonName from CodeName where Ref = " + str(ref)
    rs.execute(qry)
    comnam = rs.fetchone()[0]  # American Robin
    return comnam

def get_species():
    species_list = "species_list.txt"
    with open(species_list, 'r') as f:
        species = json.loads(f.read())
    return species

species = get_species()
print ("species_id --> verify specie count: ", len(species))

start_time = datetime.datetime.now()
print ("\nTraining Started: ", start_time )
with open(output_folder+"/res.res", "a") as res_file:
    res_file.write("Training Started ") 
    res_file.write(str(start_time))
    res_file.write('\n')
print ("class_size: ", class_size)
tx = "class_size: " + str(class_size)
with open(output_folder+"/res.res", "a") as res_file:
    res_file.write(tx + "\n")   


start_epoch = 0 # don't delete /output/model_raw.pkl if you set start_epoch > 0
N_epochs += start_epoch
print ("N_epochs: ", N_epochs)
tx = "N_epochs: " + str(N_epochs)
with open(output_folder+"/res.res", "a") as res_file:
    res_file.write(tx + "\n")   


for epoch in range(start_epoch, N_epochs): # match vs num_epochs
    # training
    torch.set_grad_enabled(True)
    test_flag=0
    CNN_net.train()
    DNN1_net.train()
    DNN2_net.train()
 
    loss_sum=0
    acc_sum=0

    for i in range(N_batches):
        [inp,lab]=create_batches_rnd(batch_size,data_folder,wav_lst_tr,snt_tr,wlen,lab_dict,0.08333) # fact_amp --> 1/12 of an octive         
        #print ("batch_size: ", batch_size, " data_folder: ", data_folder, " wlen: ", wlen)
        #exit()
        #print ("i: ", i, " inp: ", inp) # these are non zero
        poutcnn=CNN_net(inp) # these are nan
        #print ("poutcnn: ", poutcnn)
        poutdnn1=DNN1_net(poutcnn)
        pout=DNN2_net(poutdnn1)
        pred=torch.max(pout,dim=1)[1]
        loss = cost(pout, lab.long())
        acc = torch.mean((pred==lab.long()).float())

        t1 = lab.flatten().clone().detach()
        t2 = torch.argmax(pout, 1).clone().detach()
        #print ("t1 = y.flatten(): ", t1, "\n pout:", pout, "\n t2 = torch.argmax(pout, 1):", t2 )
        teq = torch.eq(t1, t2).clone().detach()
        #print("match: torch.eq(t1, t2))", teq)
        inx = 0
        for k in teq:
            if k == True:
                val = int(t1[inx].item())
                trncnt[val] +=1
                #print ("k:", k.item(), " inx:", inx, " t1[inx]:", val, "trncnt:", trncnt[val])
            inx += 1
        
        #[val,best_class]=torch.max(torch.sum(pout,dim=0),0)
        #print ("val: ", val,  " best_class: ", best_class) # these are all zero or nan
        optimizer_CNN.zero_grad()
        optimizer_DNN1.zero_grad() 
        optimizer_DNN2.zero_grad() 
    
        loss.backward()
        optimizer_CNN.step()
        optimizer_DNN1.step()
        optimizer_DNN2.step()
    
        loss_sum=loss_sum+loss.detach()
        acc_sum=acc_sum+acc.detach()
 

    loss_tot=loss_sum/N_batches
    acc_tot=acc_sum/N_batches*100

   
    # Validation
    torch.set_grad_enabled(False)
    CNN_net.eval()
    DNN1_net.eval()
    DNN2_net.eval()
    test_flag=1 
    loss_sum=0
    acc_sum=0
    acc_sum_snt=0
    #tstCnt = [0] * class_size   # number of classes

    for i in range(snt_te):
       
        [signal, fs] = sf.read(data_folder+wav_lst_te[i])

        signal=torch.from_numpy(signal).float().cuda().contiguous()
        lab_batch=lab_dict[wav_lst_te[i]]
    
        # split signals into chunks
        beg_samp=0
        end_samp=wlen
        N_fr=int((signal.shape[0]-wlen)/(wshift))
        """
        print ("snt_te:", str(snt_te))
        print ("i:", str(i))
        print ("wav_lst_te[i]:", wav_lst_te[i])        
        print ("signal.shape[0]:", str(signal.shape[0]))
        print ("wlen:", str(wlen))
        print ("wshift:", str(wshift))        
        print ("Batch_dev:", str(Batch_dev))
        print ("N_fr:", str(N_fr))
        print ("lab_batch:", str(lab_batch))
        print ("class_lay[-1]:", str(class_lay[-1]))
        """
        sig_arr=torch.zeros([Batch_dev,wlen]).float().cuda().contiguous()
        lab= Variable((torch.zeros(N_fr+1)+lab_batch).cuda().contiguous().long())
        pout=Variable(torch.zeros(N_fr+1,class_lay[-1]).float().cuda().contiguous())
        #quit()

        count_fr=0
        count_fr_tot=0
        while end_samp<signal.shape[0]:
            sig_arr[count_fr,:]=signal[beg_samp:end_samp]
            beg_samp=beg_samp+wshift
            end_samp=beg_samp+wlen
            count_fr=count_fr+1
            count_fr_tot=count_fr_tot+1
            if count_fr==Batch_dev:
                inp=Variable(sig_arr)
                pout[count_fr_tot-Batch_dev:count_fr_tot,:]=DNN2_net(DNN1_net(CNN_net(inp)))
                count_fr=0
                sig_arr=torch.zeros([Batch_dev,wlen]).float().cuda().contiguous()
   
        if count_fr>0:
            inp=Variable(sig_arr[0:count_fr])
            pout[count_fr_tot-count_fr:count_fr_tot,:]=DNN2_net(DNN1_net(CNN_net(inp)))

    
        pred=torch.max(pout,dim=1)[1]
        loss = cost(pout, lab.long())
        acc = torch.mean((pred==lab.long()).float())
        #print ("pred: ", pred, " loss: ", loss, " acc: ", acc) 
     
        t1 = lab.flatten().clone().detach()
        t2 = torch.argmax(pout, 1).clone().detach()
        #print ("t1 = y.flatten(): ", t1, "\n pout:", pout, "\n t2 = torch.argmax(pout, 1):", t2 )
        teq = torch.eq(t1, t2).clone().detach()
        #print("match: torch.eq(t1, t2))", teq)
        inx = 0
        for k in teq:
            if k == True:
                val = int(t1[inx].item())
                vldcnt[val] +=1
                #print ("k:", k.item(), " inx:", inx, " t1[inx]:", val, "vldcnt:", vldcnt[val])
            inx += 1

      
        [val,best_class]=torch.max(torch.sum(pout,dim=0),0)
        acc_sum_snt=acc_sum_snt+(best_class==lab[0]).float()
        #tstCnt[best_class] += 1
        #bc = best_class.item()
        #tstCnt[bc] += 1
    
        loss_sum=loss_sum+loss.detach()
        acc_sum=acc_sum+acc.detach()
    
    #print ("loss_sum: ", loss_sum, " snt_te: ", snt_te )
    acc_tot_dev_snt=acc_sum_snt/snt_te*100
    loss_tot_dev=loss_sum/snt_te
    acc_tot_dev=acc_sum/snt_te*100

  
    print("epoch %i, loss_tr=%f acc_tr=%6.2f loss_te=%f acc_te=%6.2f acc_te_snt=%6.2f" % (epoch, loss_tot,acc_tot,loss_tot_dev,acc_tot_dev,acc_tot_dev_snt))
    with open(output_folder+"/res.res", "a") as res_file:
        res_file.write("epoch %i, loss_tr=%f acc_tr=%6.2f loss_te=%f acc_te=%6.2f acc_te_snt=%6.2f\n" % (epoch, loss_tot,acc_tot,loss_tot_dev,acc_tot_dev,acc_tot_dev_snt))   

    calc_cntr('train', trncnt)
    calc_cntr('valid', vldcnt)


    checkpoint={'CNN_model_par': CNN_net.state_dict(),
                 'DNN1_model_par': DNN1_net.state_dict(),
                 'DNN2_model_par': DNN2_net.state_dict(),
                 }
    torch.save(checkpoint,output_folder+'/model_raw.pkl') 

    if epoch+1 == N_epochs:
        end_time = datetime.datetime.now()
        print("Training Finished: ", end_time)
        difference = end_time - start_time
        print ("difference: ", difference)
        with open(output_folder+"/res.res", "a") as res_file:
            res_file.write("Training Finished ") 
            res_file.write(str(end_time))
            res_file.write('\n')
            res_file.write("difference ") 
            res_file.write(str(difference))
            res_file.write('\n')

calc_cntr('train', trncnt)
calc_cntr('valid', vldcnt)
md.close()

