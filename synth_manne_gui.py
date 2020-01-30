from Tkinter import *
import numpy as np
import pandas as pd
import os
import librosa
import soundfile as sf
import argparse
import pyaudio
import numpy as np
from keras.layers import Input, Dense, LeakyReLU
from keras.models import Model, load_model
import tensorflow as tf 
import time
from scipy import signal 

RATE     = int(44100)
CHUNK    = int(1024)
CHANNELS = int(1)
NUM_CHUNKS = 5
ind = NUM_CHUNKS+1
proc_ind = 1
crossfade_time = int(CHUNK*3)
fade_in = np.log(np.linspace(1, 2.71, crossfade_time))
fade_out = np.log(np.linspace(2.71, 1, crossfade_time))
threshold = 1e-3
pie = np.pi
relative_height = 0.01
len_window = 4096
width_ = (len_window/2)/np.sqrt(-2*np.log(relative_height))
freq_time_ratio = -1*(pie/4)*(np.power(len_window,2)/np.log(relative_height))
last_three_der_frames = np.zeros((2049,3))
last_three_phase_frames = np.zeros((2049,3))
wn_phase = np.load('wn_phase.npy')
print('All wn phase shape')
print(wn_phase.shape)

class Application(Frame):

    global make_sine 
    def make_sine(seg_length,ii):
        global mag
        global phase 
        global remember
        global CHUNK
        global encoder
        global enc_graph
        global decoder 
        global dec_graph
        global scales  
        global recorded_scales
        global app 
        global POLL_TIME
        global RATE 
        global fade_in
        global fade_out
        global wn_phase
        global chroma_choice
        global num_latents
        global skip

        additional = 4
        enc_mag = scales*np.ones((1,num_latents))
        if skip=='skip':
            chroma_append = np.zeros((1,12))
            chroma_append[0,chroma_choice] = 1
            enc_mag = np.hstack((enc_mag,chroma_append))


        ind_array = np.arange((seg_length*ii-3),(seg_length*(ii+1)+1))
        temp_phase = wn_phase[ind_array,:]
        with dec_graph.as_default():
            temp_out_mag = decoder.predict(enc_mag)
        temp_out_mag = np.tile(temp_out_mag,(NUM_CHUNKS+additional,1))
        E = temp_out_mag*np.exp(1j*temp_phase)
        _, now_out = signal.istft(E.T, fs=44100, noverlap=3*1024, nfft=4096)
        out = np.float32(now_out[3*CHUNK:]*(0.08/np.max(np.abs(now_out))))
        final_out = out.reshape(((len(out)/CHUNK),CHUNK))
        return final_out

    global callback 
    def callback(in_data, frame_count, time_info, status):
        global ind
        global proc_ind 
        global NUM_CHUNKS
        global all_data

        if ind>=(NUM_CHUNKS-1):
            all_data = make_sine(NUM_CHUNKS,proc_ind) 
            ind = 0
            proc_ind+=1
        data = all_data[ind,:] #Send a chunk to the audio buffer when it asks for one
        ind +=1 
        return (data, pyaudio.paContinue)

    def render(self):
        global mag
        global phase 
        global remember
        global CHUNK
        global encoder
        global enc_graph
        global decoder 
        global dec_graph
        global scales  
        global app 
        global recorded_scales

        print(scales)
        ind_array = np.arange((1000),(1000*(2)))
        temp_phase = wn_phase[ind_array,:]
        enc_mag = scales*np.ones((1,15))
        with dec_graph.as_default():
            temp_out_mag = decoder.predict(enc_mag)
        temp_out_mag = np.tile(temp_out_mag,(200,1))
        E = temp_out_mag*np.exp(1j*temp_phase)
        _, now_out = signal.istft(E.T, fs=44100, noverlap=3*1024, nfft=4096)
        out = np.float32(now_out[3*CHUNK:]*(0.08/np.max(np.abs(now_out))))

        sf.write('rendered.wav', out, 44100, subtype='PCM_16')
        print('done rendering')

    def record(self):
        global mag
        global phase 
        global remember
        global CHUNK
        global encoder
        global enc_graph
        global decoder 
        global dec_graph
        global scales  
        global app 
        global recorded_scales
        global proc_ind

        first_ind = 0
        last_ind = 0
        total_frames = 0
        if self.RECORD_var.get() == 1:
            first_ind = proc_ind
            print('Button On')
            self.start_net()
        else:
            last_ind = proc_ind
            print('Button off')
            self.pause_sounds()

            total_frames = (last_ind-first_ind)*NUM_CHUNKS
            out_scales = np.ones((total_frames,15))
            temp_scales = np.vstack(recorded_scales)
            a = temp_scales.shape[0]
            increase_by = total_frames//a+1
            kurt=0
            for ii in range(a):
                the_rows = np.arange((kurt*increase_by),min(((kurt+1)*increase_by),total_frames))
                out_scales[the_rows,:] = np.tile(temp_scales[ii,:],(len(the_rows),1))
                kurt+=1
            ind_array = np.arange((first_ind),(NUM_CHUNKS*(last_ind)))
            temp_phase = wn_phase[ind_array,:]

            with dec_graph.as_default():
                temp_out_mag = decoder.predict(out_scales)

            E = temp_out_mag*np.exp(1j*temp_phase)
            _, now_out = signal.istft(E.T, fs=44100, noverlap=3*1024, nfft=4096)
            out = np.float32(now_out[3*CHUNK:]*(0.8/np.max(np.abs(now_out))))

            sf.write('recorded.wav', out, 44100, subtype='PCM_16')
            print('done recording')



    def model_to_mem(self):
        global decoder 
        global dec_graph

        data_path = os.path.join(os.getcwd(),self.model_name.get()+'_trained_decoder.h5')
        decoder = load_model(data_path, compile=False)
        decoder._make_predict_function()
        dec_graph = tf.get_default_graph()

    def process_track(self):
        global mag
        global phase
        global remember

        len_window = 4096 #Specified length of analysis window
        hop_length_ = 1024 #Specified percentage hop length between windows

        filename_in = self.track_name.get()
        data_path = os.path.join(os.getcwd(),filename_in)
        y, sr = librosa.load(data_path, sr=44100, mono=True)

        D = librosa.stft(y,n_fft=len_window, window='hann')
        mag = D 
        mag = np.abs(mag) #Magnitude response of the STFT
        remember = mag.max(axis=0)+0.000000001 #Used for normalizing STFT frames (with addition to avoid division by zero)
        mag = mag / remember #Normalizing
        phase = np.angle(D) #Phase response of STFT
        mag = mag.T

        return mag, phase, remember  



    def start_net(self):
        global p 
        global stream
        self.model_to_mem()

        p = pyaudio.PyAudio()

        stream = p.open(format=pyaudio.paFloat32,
                        channels=CHANNELS,
                        frames_per_buffer=CHUNK,
                        rate=RATE,
                        output=True,
                        stream_callback=callback)


        stream.start_stream()
        time.sleep(0.1)

    def pause_sounds(self):
        global p 
        global stream
        global ind
        global proc_ind 
        
        stream.stop_stream()
        print('sounds paused')
        stream.close()
        p.terminate()
        ind = NUM_CHUNKS+1
        proc_ind = 0

    def quit(self):
        root.destroy()
        

    def createWidgets(self):
        self.QUIT = Button(self)
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"]   = "red"
        self.QUIT["command"] =  self.quit
        self.QUIT.pack()
        self.QUIT.place(relx=0.45,rely=0.9)

        self.model_name = Entry(self)
        self.model_name.pack()
        self.model_name.place(relx=0.4,rely=0.65)
        self.label_1 = Label(self,text='Model Name')
        self.label_1.pack()
        self.label_1.place(relx=0.25,rely=0.65)

        self.START = Button(self)
        self.START["text"] = "START"
        self.START["fg"]   = "green"
        self.START["command"] =  lambda: self.start_net()
        self.START.pack()
        self.START.place(relx=0.45,rely=0.85)

        self.PAUSE = Button(self)
        self.PAUSE["text"] = "PAUSE"
        self.PAUSE["fg"]   = "black"
        self.PAUSE["command"] =  lambda: self.pause_sounds()
        self.PAUSE.pack()
        self.PAUSE.place(relx=0.45,rely=0.8)

        self.RECORD_var = IntVar()
        self.RECORD = Checkbutton(self, variable=self.RECORD_var)
        self.RECORD["text"] = "RECORD"
        self.RECORD["fg"]   = "black"
        self.RECORD["command"] =  lambda: self.record()
        self.RECORD.pack()
        self.RECORD.place(relx=0.45,rely=0.75)

        self.RENDER = Button(self)
        self.RENDER["text"] = "RENDER"
        self.RENDER["fg"]   = "black"
        self.RENDER["command"] =  lambda: self.render()
        self.RENDER.pack()
        self.RENDER.place(relx=0.45,rely=0.7)



    def createSliders(self):
        global scales 
        global num_latents
        scales = np.ones(num_latents)
        self.scale_list = []
        for w in range(num_latents):
            scale = Scale(self,from_=110, to=-10,length=200)
            scale.pack()
            scale.place(relx=w/(float(num_latents)),rely=0.2)
            scale.set(0)
            scales[w]=scale.get()
            self.scale_list.append(scale)

    def createButtons(self):
        global chroma_val
        self.chroma_val = IntVar()
        self.chroma_val.set(0)
        NOTE_OPTIONS = [
        ('C',0),
        ('C#',1),
        ('D',2),
        ('D#',3),
        ('E',4),
        ('F',5),
        ('F#',6),
        ('G',7),
        ('G#',8),
        ('A',9),
        ('A#',10),
        ('B',11)
        ]

        for text, val in NOTE_OPTIONS:
            b = Radiobutton(self, text=text, value=val, variable=self.chroma_val)
            b.pack()
            b.place(relx=0.2+val/19.,rely=0.1)


    def update_scales(self):
        global scales 
        global recorded_scales
        global POLL_TIME
        global chroma_choice
        global num_latents

        POLL_TIME = 100
        chroma_choice = self.chroma_val.get()
        temp_scales = np.ones(8)
        for w in range(num_latents):
            temp_scales[w]=self.scale_list[w].get()/300.
        scales = temp_scales
        if self.RECORD_var.get() == 1:
            recorded_scales.append(scales)
        self.after(POLL_TIME, self.update_scales)


    def __init__(self, master=None):
        global recorded_scales

        Frame.__init__(self, master,width=800, height=800)
        self.pack()
        self.createWidgets()
        self.createButtons()
        self.createSliders()
        recorded_scales = []
        self.update_scales()

global app 
global num_latents
global skip
num_latents = int(sys.argv[1])
skip = sys.argv[2]
root = Tk()
app = Application(master=root)
app.mainloop()
root.destroy()