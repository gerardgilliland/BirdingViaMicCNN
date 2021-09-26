# list_to_dict.py
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import shutil
import pickle
import ast
import json

all_list = './species/all_list.txt'
species_list = 'species_list.txt'
class_dict_file = 'species_lab.npy'
text_dict = 'check_dict.txt'

def get_species():
    with open(species_list, 'r') as f:
        species = json.loads(f.read())
    return species

species = get_species()
print (species)
print ("-----------------------")
print ("species count: ", len(species))

fndata = "{"

fn = open (all_list, "r")
for line in fn:
    inx = line.find("/")
    specie = line[:inx] 
    label = species.index(specie)  
    #print ("line: ", line, " specie: ", specie, " label: ", label)
    fndata += "'" + line[:-1] + "': " + str(label) + ", "

fn.close
fndata += "}"

fck = open (text_dict, "w")
fck.write(fndata)
fck.close

dictionary = ast.literal_eval(fndata)
lab_dict=np.save(class_dict_file, dictionary)


