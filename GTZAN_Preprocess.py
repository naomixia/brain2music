
import os
import scipy.io
from scipy.io.wavfile import read, write
from scipy.io import loadmat
import numpy as np
import glob
import wave
from scipy import fromstring, int16
import math
import glob
import shutil
import opendatasets as od
od.download("https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification")


cwd = os.getcwd()
load_folder = os.path.join(cwd, "gtzan-dataset-music-genre-classification/Data/genres_original")
genre_list = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
wg_order = loadmat(os.path.join(cwd, "WithinGenre_Order.mat"))["within_genre_rn"]
chosen_songs = loadmat(os.path.join(cwd, "ChosenSongs_raw.mat"))["Tarray"]
ag_order_trn = loadmat(os.path.join(cwd, "AcrossGenre_Order_Trn.mat"))["across_genre_rn_trn"]
ag_order_test = loadmat(os.path.join(cwd, "AcrossGenre_Order_Test.mat"))["across_genre_rn_test"]
start_data = loadmat(os.path.join(cwd, "StartData_raw.mat"))["StartData"]
end_data = loadmat(os.path.join(cwd, "EndData_raw.mat"))["EndData"]
rms_mean = loadmat(os.path.join(cwd, "Mean_RMS.mat"))["RMSmean"]
save_folder = os.path.join(cwd, "genres_preproc")
os.mkdir(save_folder)

stim_size = 15  #final sound length(s)
fs = 22050 #Sampling rate
rise_len = 2  #sound rising length 
n_genre = 10
n_token = 54

t_wav_data = np.zeros([n_genre, n_token, stim_size*fs])
for genre_id in range(n_genre):
    for token_id in range(n_token):
        #Load raw signal
        targ_song = np.sort(chosen_songs[genre_id, :] - 1)[wg_order[token_id, genre_id] - 1]
        sort_ind = np.argsort(chosen_songs[genre_id, :] - 1)
        load_name = os.path.join(load_folder, genre_list[genre_id], genre_list[genre_id] + ".000{:02d}.wav".format(targ_song))
        fss, wav_data = scipy.io.wavfile.read(filename=load_name)

        #Cut into 15s
        start_point_sort = start_data[genre_id, sort_ind]
        start_point = start_point_sort[wg_order[token_id, genre_id] - 1] - 1
        end_point_sort = end_data[genre_id, sort_ind]
        end_point = end_point_sort[wg_order[token_id, genre_id] - 1] - 1
        wav_data = wav_data[start_point:end_point+1]

        #Apply onset and ofset form
        rise_len_fs = rise_len * fs
        rise_data = np.sin(np.linspace(0, math.pi/2, rise_len_fs))
        mask = np.concatenate([rise_data, np.ones([np.round(fs * stim_size) - rise_len_fs*2, ]), np.flipud(rise_data)])
        wav_data = wav_data * mask
        
        #RMS normalization
        rms = np.sqrt(np.power(wav_data, 2).mean())
        wav_data = wav_data * rms_mean / rms 
        
        t_wav_data[genre_id, token_id, :] = wav_data


#Save training data
for run in range(12):
    for block in range(4):
        for trial in range(10):
            genre_id = ag_order_trn[run, block, trial] - 1
            wav_data = t_wav_data[genre_id, 4*run + block, :]
            save_fname = os.path.join(save_folder, "Stim_Training_Run{:02d}_{:02d}_{}.wav".format(run+1, 10*block+trial+1, genre_list[genre_id]))
            scipy.io.wavfile.write(filename=save_fname, rate=fs, data=wav_data)


#Present the same stimuli at the end of each run and at the begining of the next run
#for run in range(12):
#    cp_from = glob.glob(os.path.join(save_folder, "Stim_Training_Run{:02d}_40*.wav".format(run+1)))[0]
#    genre_name = cp_from.split('_')[-1][0:-4]
#    if run == 11:
#        cp_to = os.path.join(save_folder, "Stim_Training_Run01_00_{}.wav".format(genre_name))    
#    else:
#        cp_to = os.path.join(save_folder, "Stim_Training_Run{:02d}_00_{}.wav".format(run+2, genre_name))
#    shutil.copyfile(cp_from, cp_to)



#Change stimulus order manually
os.rename(os.path.join(save_folder, "Stim_Training_Run02_01_metal.wav"), os.path.join(save_folder, "Stim_Training_Run02_02_metal.wav"))
os.rename(os.path.join(save_folder, "Stim_Training_Run02_02_rock.wav"), os.path.join(save_folder, "Stim_Training_Run02_01_rock.wav"))

os.rename(os.path.join(save_folder, "Stim_Training_Run03_01_pop.wav"), os.path.join(save_folder, "Stim_Training_Run03_02_pop.wav"))
os.rename(os.path.join(save_folder, "Stim_Training_Run03_02_metal.wav"), os.path.join(save_folder, "Stim_Training_Run03_01_metal.wav"))

os.rename(os.path.join(save_folder, "Stim_Training_Run09_01_blues.wav"), os.path.join(save_folder, "Stim_Training_Run09_02_blues.wav"))
os.rename(os.path.join(save_folder, "Stim_Training_Run09_02_jazz.wav"), os.path.join(save_folder, "Stim_Training_Run09_01_jazz.wav"))

os.rename(os.path.join(save_folder, "Stim_Training_Run11_01_classical.wav"), os.path.join(save_folder, "Stim_Training_Run11_02_classical.wav"))
os.rename(os.path.join(save_folder, "Stim_Training_Run11_02_blues.wav"), os.path.join(save_folder, "Stim_Training_Run11_01_blues.wav"))

os.rename(os.path.join(save_folder, "Stim_Training_Run12_01_reggae.wav"), os.path.join(save_folder, "Stim_Training_Run12_02_reggae.wav"))
os.rename(os.path.join(save_folder, "Stim_Training_Run12_02_classical.wav"), os.path.join(save_folder, "Stim_Training_Run12_01_classical.wav"))

os.rename(os.path.join(save_folder, "Stim_Training_Run03_31_jazz.wav"), os.path.join(save_folder, "Stim_Training_Run03_32_jazz.wav"))
os.rename(os.path.join(save_folder, "Stim_Training_Run03_32_hiphop.wav"), os.path.join(save_folder, "Stim_Training_Run03_31_hiphop.wav"))

os.rename(os.path.join(save_folder, "Stim_Training_Run08_11_classical.wav"), os.path.join(save_folder, "Stim_Training_Run08_12_classical.wav"))
os.rename(os.path.join(save_folder, "Stim_Training_Run08_12_disco.wav"), os.path.join(save_folder, "Stim_Training_Run08_11_disco.wav"))

os.rename(os.path.join(save_folder, "Stim_Training_Run11_31_classical.wav"), os.path.join(save_folder, "Stim_Training_Run11_32_classical.wav"))
os.rename(os.path.join(save_folder, "Stim_Training_Run11_32_country.wav"), os.path.join(save_folder, "Stim_Training_Run11_31_country.wav"))


#Save training data
for run in range(6):
    for block in range(4):
        for trial in range(10):
            genre_id = ag_order_test[run, block, trial] - 1
            wav_data = t_wav_data[genre_id, 48 + run, :]
            save_fname = os.path.join(save_folder, "Stim_Test_Run{:02d}_{:02d}_{}.wav".format(run+1, 10*block+trial+1, genre_list[genre_id]))
            scipy.io.wavfile.write(filename=save_fname, rate=fs, data=wav_data)


#Present the same stimuli at the end of each run and at the begining of the same run
#for run in range(6):
#    cp_from = glob.glob(os.path.join(save_folder, "Stim_Test_Run{:02d}_40*.wav".format(run+1)))[0]
#    genre_name = cp_from.split('_')[-1][0:-4]
#    cp_to = os.path.join(save_folder, "Stim_Test_Run{:02d}_00_{}.wav".format(run+1, genre_name))
#    shutil.copyfile(cp_from, cp_to)
