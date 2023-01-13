import torch
from pprint import pprint
import librosa
import librosa.display
import os
import sys
sys.path.insert(1, '/u/home/a/asarkar/project-jflint/Language_diarization/BUT/')
import bottleneck2posterior
import audio2bottleneck
import uvector
from data_load import prepare_data, create_uvector
from sklearn.metrics import confusion_matrix
import seaborn as sns

class vad_data:
    def __init__(self, fname):
        self.fname = fname
        
    def make_chunks(self):
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=True)
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        for folder in os.listdir(self.fname):
            for file in os.listdir(os.path.join(self.fname,folder)):
                pth = os.path.join(self.fname,folder,file)
                print(pth)
                wav = read_audio(pth,sampling_rate=8000)
                speech_timestamps = get_speech_timestamps(wav, model,sampling_rate=8000)

                for i in range(len(speech_timestamps)):
                    t = '/u/home/a/asarkar/scratch/vad_test/'
                    save_audio(t+file.split('.')[0]+'_'+str(i+1)+'.wav', collect_chunks([speech_timestamps[i]],wav), sampling_rate=8000)
            
    def predict_chunks(self):
#         self.make_chunks()
        
#         if os.path.exists(os.path.join('/u/home/a/asarkar/scratch/chunks_results/','MFCC')):
#             pass
#         else:
#             os.mkdir(os.path.join('/u/home/a/asarkar/scratch/chunks_results/','MFCC'))
        
#         if os.path.exists(os.path.join('/u/home/a/asarkar/scratch/chunks_results/','BNF')):
#             pass
#         else:
#             os.mkdir(os.path.join('/u/home/a/asarkar/scratch/chunks_results/','BNF'))
            
#         for i in os.listdir('/u/home/a/asarkar/scratch/vad_test/'):
#             pth = '/u/home/a/asarkar/scratch/vad_test/' + i
#             x,sr = librosa.load(pth,sr=8000)

#             if os.path.isdir(pth) == False:
#                 create_mfcc(pth,x,sr).mfcc()
#                 create_bnf(pth,"BabelMulti").bnf()
#             else:
#                 pass

        Tru1, Pred1 = create_uvector('/u/home/a/asarkar/scratch/BNF_Data_split1/','bnf').pred_bilstm()
#         Tru0, Pred0 = create_uvector('/u/home/a/asarkar/scratch/chunks_results/MFCC/','mfcc').pred_bilstm()
#         Tru2, Pred2 = create_xvector('/u/home/a/asarkar/scratch/chunks_results/MFCC/','mfcc').pred_tdnn()
#         Tru3, Pred3 = create_xvector('/u/home/a/asarkar/scratch/chunks_results/BNF/','bnf').pred_tdnn()

        for i in range(Tru1.shape[0]):
            print(Tru1[i], Pred1[i])
            
#         t = [Tru0, Pred0, Tru2, Pred2]
#         return t
    
    def plot_results(self):
        t = self.predict_chunks()
        fig, axs = plt.subplots(2,2,figsize=(10, 6))
        i=0
        k=[0,0,1,1]
        t=[0,1,0,1]
        while(i<7):
            confusion_matrix(t[i], t[i+1])
            sns.heatmap(ax=axes[k[j],t[j]], x=t[i], y=t[i+1],annot=True, annot_kws={"size": 16})
            i=i+2
            j=j+1
           

if __name__ == "__main__":
#     audio_path = "/u/home/a/asarkar/scratch/LID_data/"
#     nn = "BabelMulti"
#     new_path = prepare_data(audio_path).split_files()
#     print("data prepared")
#     new_path = new_path+'/train/'
    vad = vad_data('new_path')
    print("vad initialised")
    t = vad.predict_chunks()