def main(wavName, db_file, cnnModelName, speciesLab, speciesList, speciesConfig):
    result = ""
    result += "wavName: " + wavName + "\n"
    #result += "db_file: " + db_file + "\n"
    #result += "cnnModelName: " + cnnModelName + "\n"
    #result += "speciesLab: " + speciesLab + "\n"
    #result += "speciesList: " + speciesList + "\n"
    #result += "speciesConfig: " + speciesConfig + "\n"

    import os
    import librosa
    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    torchversion = torch.__version__
    result += "torchversion: " + torchversion + "\n"

    import sys
    import numpy as np
    numpyversion = np.__version__
    result += "npversion: " + numpyversion + "\n"

    #import pyaudio
    from dnn_models import MLP
    from dnn_models import SincNet as CNN
    from data_io import ReadList,read_conf,str_to_bool
	
    import json
    import time
    import sqlite3
    sqliteVersion = sqlite3.version
    result +=  "sqliteVersion: " + sqliteVersion + "\n"
    link = sqlite3.connect(db_file)
    HOP_LENGTH = 512

    import datetime
    start_time = datetime.datetime.now()
    result += "start_time: " + str(start_time) + "\n"

    # Reading cfg file (in data_io.py)
    options=read_conf(speciesConfig)
    result += "return with options from read_conf" + "\n"

    #[data] -- THIS POINTS TO A DIFFERENT DATASET THAN CFG FILE
    db = 20 # only change this once per run. I might decide 32db is better but then all should be 32db
    data_folder = "./Song"

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

    result += "fs: " + str(fs) + " cw_len: " + str(cw_len) + "\n"

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

    result += "wlen: " + str(wlen) + "\n"

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
    lab_dict=np.load(speciesLab,allow_pickle=True).item()
    result += "Load label dictionary\n"

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
                #print ("class: ", t, " count: ", totcount[t], com)
            if maxcount < totcount[t]:
                maxcount = totcount[t]
                bestclass = t

        specie = species[bestclass]
        #print ("specie: ", specie)
        ref = specie[1:]
        #print ("ref: ", ref)
        comnam = ConvertToName(ref)
        #print ("calc_cntr: ", bestclass, " maxcount: ", maxcount, " ref: ", ref, comnam)
        return bestclass, maxcount

    def get_species():
        with open(speciesList, 'r') as f:
            species = json.loads(f.read())
        return species

    species = get_species()
    result += "loaded species -- len: " + str(len(species)) + "\n"

    def norm(data):
        return librosa.util.normalize(data, np.inf, axis=0, threshold=None, fill=None)

    # if model exists and is saved
    if os.path.isfile(cnnModelName):
        checkpoint = torch.load(cnnModelName, map_location='cpu')
        CNN_net.load_state_dict(checkpoint['CNN_model_par'])
        DNN1_net.load_state_dict(checkpoint['DNN1_model_par'])
        DNN2_net.load_state_dict(checkpoint['DNN2_model_par'])
        result += "model_raw.pkl loaded" + "\n"
    else:
        #print ("model_raw.pkl is missing")
        result += "model_raw.pkl is missing" + "\n"
        exit()

    # Validation

    CNN_net.eval()
    DNN1_net.eval()
    DNN2_net.eval()
    test_flag=1
    loss_sum=0
    acc_sum=0
    acc_sum_snt=0
    class_size = int(class_lay[0])
    totcount = [0] * class_size

    torch.no_grad()  # don't mess with the model just use it to predict the bird

    inx = wavName.find("Song/") + 5
    random_specie = wavName[inx:]
    inx = random_specie.find("_")
    random_specie = random_specie[:inx]
    result += "random_specie: " +  random_specie + "\n"

    audio, sr = librosa.load(wavName)
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
    result += "wlen: " + str(wlen) + " wshift:" + str(wshift) + "\n"
    result += "this_len: " + str(this_len) +  " number frames: " +  str(N_fr) + "\n"

    sig_arr=torch.zeros([Batch_dev,wlen]).float().cpu().contiguous()
    lab= Variable((torch.zeros(N_fr)).cpu().contiguous().long())
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
            result += str(count_fr) + " frames loaded into model inp" + "\n"
            inp=Variable(sig_arr)
            pout[count_fr_tot-Batch_dev:count_fr_tot,:]=DNN2_net(DNN1_net(CNN_net(inp)))
            count_fr=0
            sig_arr=torch.zeros([Batch_dev,wlen]).float().cpu().contiguous()

    if count_fr>0:
        result += str(count_fr) + " frames loaded into model inp" + "\n"
        inp=Variable(sig_arr[0:count_fr])
        pout[count_fr_tot-count_fr:count_fr_tot,:]=DNN2_net(DNN1_net(CNN_net(inp)))

    pred=torch.max(pout,dim=1)[1]
    loss = cost(pout, lab.long())
    acc = torch.mean((pred==lab.long()).float())

    t1 = lab.flatten().clone().detach()
    t2 = torch.argmax(pout, 1).clone().detach()

    result_t2 = t2.cpu()
    bestclass, maxcount = calc_cntr('valid', result_t2)

    loss_sum=loss_sum+loss.detach()
    acc_sum=acc_sum+acc.detach()

    acc_tot_dev_snt=acc_sum_snt/N_fr*100
    loss_tot_dev=loss_sum/N_fr
    acc_tot_dev=acc_sum/N_fr*100

    specie = species[bestclass]
    ref = specie[1:]
    comnam = ConvertToName(ref)

    if comnam == random_specie:
        prediction = "successful."
    else:
        prediction = "fails."
    result += "predict: " + str(bestclass) + " ref: " + str(ref) + " " + comnam + " " + prediction + "\n"

    x = maxcount/N_fr * 100
    result += "Percent accuracy " + str(maxcount) + "/" + str(N_fr) +" = {:.2f}".format(x) + " %" + "\n"

    end_time = datetime.datetime.now()
    result += "Prediction Finished: " + str(end_time) + "\n"
    difference = end_time - start_time
    result += "prediction time: " + str(difference) + "\n"

    return result

