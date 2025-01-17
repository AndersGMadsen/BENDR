import os
if not os.getcwd().endswith('BENDR'): os.chdir(os.path.dirname(os.getcwd()))
from mne.io import read_raw_edf
import mne, torch, re, warnings
from numpy import empty
import pandas as pd
import matplotlib.pyplot as plt
from BENDR.dn3_ext import LinearHeadBENDR
import torch
import mne
from dn3.configuratron.config import RawOnTheFlyRecording, RawTorchRecording, _DumbNamespace
from dn3.transforms.instance import MappingDeep1010, TemporalInterpolation
import numpy as np
from dn3.transforms.instance import To1020
from BENDR.utils import get_ds
import pickle
import random
from tqdm import tqdm

# renames TUH channels to conventional 10-20 system
def TUH_rename_ch(MNE_raw=False):
    # MNE_raw
    # mne.channels.rename_channels(MNE_raw.info, {"PHOTIC-REF": "PROTIC"})
    for i in MNE_raw.info["ch_names"]:
        reSTR = r"(?<=EEG )(.*)(?=-)" # working reSTR = r"(?<=EEG )(.*)(?=-REF)"
        reLowC = ['FP1', 'FP2', 'FZ', 'CZ', 'PZ']

        if re.search(reSTR, i) and re.search(reSTR, i).group() in reLowC:
            lowC = i[0:5]+i[5].lower()+i[6:]
            mne.channels.rename_channels(MNE_raw.info, {i: re.findall(reSTR, lowC)[0]})
        elif i == "PHOTIC-REF":
            mne.channels.rename_channels(MNE_raw.info, {i: "PHOTIC"})
        elif re.search(reSTR, i):
            mne.channels.rename_channels(MNE_raw.info, {i: re.findall(reSTR, i)[0]})
        else:
            continue
            # print(i)
    TUH_pick = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Cz']  # A1, A2 removed
    MNE_raw.pick_channels(ch_names=TUH_pick)
    MNE_raw.reorder_channels(TUH_pick)
    #print(MNE_raw.info["ch_names"])
    return MNE_raw

# create folders 
# based on concepts and targets
def create_dir(path, label):
    #creates folders / paths BASED ON LABELS EG, CONCEPTS AND TARGETS 
    # return path name
    parent_dir = path
    path = os.path.join(parent_dir, label)

    if not(os.path.exists(path)) :
        os.mkdir(path)
        print("Directory '% s' created" % path)

    return path
    #return path + '/' + label
    
    #if exists then 

# find inverse time windows 
def find_inverse_time_window(tlen, annos4session):
    """ 
    Finds time window of EEG reecording BETWEEN annotations based on the end time of annotation
    returns: time window start and end if annotation is found else 0
    """
    tstart = 0
    tend = 0

    for i in range(0,len(annos4session) -1):
        t1 =  float(annos4session[2].iloc[i]) #get endtime of annotation i
        t2 = float(annos4session[1].iloc[i+1]) #get starttime of annotation i+1
        twanted = t2-t1
        if twanted > tlen: #check if time-window is large enough based on tlen
            tm = t1+twanted/2 #find middle of time-window
            tstart = tm-tlen/2
            tend = tm+tlen/2
            break
    return tstart, tend
# find time windows 

def find_time_window(tlen, annos4session):
    tstart = 0
    tend = 0

    for i in range(0,len(annos4session)):
        t1 =  float(annos4session[1].iloc[i])
        t2 = float(annos4session[2].iloc[i])
        twanted = t2-t1
        if twanted > tlen:
            tm = t1+twanted/2
            tstart = tm-tlen/2
            tend = tm+tlen/2
            break
    return tstart, tend

def get_and_transform_data(filepath, t1, t2):
    """ RETURNS A TENSOR SIZE [1, CHANNELS = 20 , SAMPLES]
        Which represents one example of a concept
     """
    # read in files and select time frame
    raw = read_raw_edf(filepath).crop(t1,t2)
    # rename channels 
    TUH_rename_ch(raw)
    # set param 
    sfreq = 250
    new_sfreq = 256
    data_max = 3276.7
    data_min = -1583.9258304722666
    tlen = t2 - t1    
    #print(tlen)
    
    # use Raw Torch recording to set recording to torch object according to tlen
    recording = RawTorchRecording(raw, tlen , stride=1, decimate=1, ch_ind_picks=None, bad_spans=None)

    # Transform data using Mapping
    _dum = _DumbNamespace(dict(channels=recording.channels, info=dict(data_max=data_max,
                                                                                data_min=data_min)))
    xform = MappingDeep1010(_dum, return_mask = True)

    #For each item retrieved by __getitem__, transform is called to modify that item.
    recording.add_transform(xform)

    # reset sampling frequency 
    if sfreq != new_sfreq:
            new_sequence_len = int(tlen * new_sfreq) 
            recording.add_transform(TemporalInterpolation(new_sequence_len, new_sfreq=new_sfreq))

    # transform  using 1020
    recording.add_transform(To1020())
    # add 1 dim to Tnsor for epoch thing... 
    output1020 = recording.__getitem__(0)[0]
    final_example = output1020[None, :]
    # RETURNS A TENSOR SIZE [1, CHANNELS = 20 , SAMPLES]
    return final_example




def populate_folders(savePath, DictOfRawFiles, csvPath,  label, tlen, numExp):
    
    #create csv anno dataframe 
    dfannos = pd.read_csv(csvPath, sep=",", skiprows=6, header=None)

    #Get a random file name to cut out concept
    GetRandomFileNr = random.randint(0,len(DictOfRawFiles) -1)
    randomPatientInfo = DictOfRawFiles[GetRandomFileNr]


    # genertate list of sessions filenames (or first numexp to start with )

    
    # returns file name and filepath
    counter_examples = 0

    while counter_examples < numExp  :  #(loop through file names and get 10 samples)
        #find time windows 
        # get twindow
        filename = randomPatientInfo['filename']
        filepath = randomPatientInfo['path']

        annos4session = pd.read_csv(filepath.replace('.edf', '.csv'), sep=",", skiprows=7, header=None)

        if label == "random":
            
            t1, t2 = find_inverse_time_window(tlen, annos4session)
        else:    
            annos4session = annos4session.loc[lambda df: df[3] == label, :]
            t1, t2 = find_time_window(tlen, annos4session)
        
        #check if timewindow exists that is greater than 0
        if not (annos4session.empty or t1+t2==0): 
            #get EEG recording on format ready for BENDR 
            final_example = get_and_transform_data(filepath, t1, t2)
            counter_examples += 1
            print(counter_examples)

            # save to folder based on label 
            pickleName = filename + str(counter_examples) + '.pkl'
            picklePath = os.path.join(savePath, pickleName)

            # save pickle
            with open(picklePath, 'wb') as handle:
                pickle.dump(final_example, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #Check next file randomly
        GetRandomFileNr = random.randint(0,len(DictOfRawFiles) -1)
        # print("Random file:")
        # print(GetRandomFileNr)
        randomPatientInfo = DictOfRawFiles[GetRandomFileNr]

def getDictOfFilesFromTUAR(path) : 
    EEG_count = 0
    EEG_dict = {}

    print("PATH:")
    print(path)
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if f.endswith(".edf")]:
            """For every edf file found somewhere in the directory, it is assumed the folders hold the structure: 
            ".../id/patientId/sessionId/edfFile".
            Therefore the path is split backwards and the EEG_dict updated with the found ids/paths.
            Furthermore it is expected that a csv file will always be found in the directory."""
            session_path_split = os.path.split(dirpath)
            patient_path_split = os.path.split(session_path_split[0])
            id_path_split = os.path.split(patient_path_split[0])

            EEG_dict.update({EEG_count: {"id": id_path_split[1],
                                            "patient_id": patient_path_split[1],
                                            "session": session_path_split[1],
                                            "path": os.path.join(dirpath, filename),
                                            "filename": os.path.splitext(filename)[0]}})
            #new_index_patient = pd.DataFrame({'index': EEG_count,'patient_id': EEG_dict[EEG_count]["patient_id"], 'window_count' : 0, 'elec_count' : 0}, index = [EEG_count])
            
            EEG_count += 1
    
    return EEG_dict


# create random files if do not exist
def create_Random_files(numRand, path, numExp, DictOfRawFiles, tlen):
    randomPath = 'random500_' 
    label = "random"
    for i in tqdm(range(0,numRand)): 
        savePath =  create_dir(path, randomPath + str(i))# creates numRand folders with FOLDERS SHOULD BE CALLED random_500_{i}
        # while counter_examples < numExp
        populate_folders(savePath, DictOfRawFiles, csvPath,  label, tlen, numExp)
        # use time window to find where we do NOT have an annotation .... USE ABOVE
    pass 

target = ['left']
concepts = ['eyem', 'musc'] #['eyem', 'musc'] #the labels that you want as concepts
# structore of TUAR data : 
# TUAR\edf\montage\patient\session\recording.edf
dataPath = 'data/datasets/tuh_eeg_artifact/v3.0.0/edf/01_tcp_ar' #where raw data is stored 
mainPath = 'data/datasets/random' #where we want to save folders 
# structure of csv labels: csv table with key, channel_label, start_time, stop_time, artifact_label
csvPath = "data/labels_01_tcp_ar.csv" #where csv file is stored
numRand = 50 # how many random files do we want in each folder/experiment
numExp = 30 #hoq many folders/experiments


# # Run this to create a dictionary that holds the structure of the TUAR files
dictOfTuarFiles = getDictOfFilesFromTUAR(dataPath)
#print(dictOfTuarFiles)
# # save pickle
# with open("TUARDICT.pkl", 'wb') as handle:
#     pickle.dump(dictOfTuarFiles, handle, protocol=pickle.HIGHEST_PROTOCOL)

# '''
# Use this to read in the dictionary once you have created it 
# #read pickle: 
# #file = open("TUARDICT.pkl",'rb')
# #dictOfTuarFiles = pickle.load(file)
# '''


# '''
# Creates and saves pickles of labeled concepts or random concepts 
# '''

# #Select which concepts to create
runConcepts = True  
runRamdoms = False  # @ WILLIAM & ANDERS, TO CREATE RANDOM FILES, SET TRUE

tlen = 4  # length of concept


# # FOR CONCEPTS 

if(runConcepts == True) : 
    for concept in concepts:
        savePath =  create_dir(mainPath, concept) #create directory for label in path 
        populate_folders(savePath, dictOfTuarFiles, csvPath,  concept, tlen, numExp)

# FOR RANDOM FILES ---- @ WILLIAM & ANDERS, TO CREATE RANDOM FILES

# if(runRamdoms== True) : 
#     create_Random_files(numRand, mainPath, numExp, dictOfTuarFiles, tlen)
