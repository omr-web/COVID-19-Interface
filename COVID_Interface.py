# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:52:22 2022
Copyright (C) 2022 "Omer CEBECI
@author: OMER CEBECI 
"""

## KÜTÜPHANELERİN EKLENMESİ
import os
import glob
import numpy as np
from datetime import datetime
from os import scandir
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy.io.wavfile import read 
from scipy import signal
from IPython.lib.display import Audio
import scipy.signal as sgnl
import pandas as pd
import mutagen
from mutagen.wave import WAVE
import librosa
import pylab
import sklearn.metrics as metrics
import python_speech_features as psf
from matplotlib import cm
import python_speech_features
from matplotlib import cm
###############################
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

from PIL import ImageTk,Image

import pickle


##############################################################################
################################################################################
window=tk.Tk()
window.geometry("1080x640")
window.wm_title("COVID-19 TANI KONMA PROGRAMI")

##global variables
record_name=""
count=0
record_max_mfcc=np.arange(1*13,dtype = float).reshape(1,13)
record_max_mel=np.arange(1*128,dtype = float).reshape(1,128) 
record_max_mfcc_delta=np.arange(1*13,dtype = float).reshape(1,13)

frame_left=tk.Frame(window,width=540,height=640,bd="2")
frame_left.grid(row=0,column=0)


frame_right=tk.Frame(window,width=540,height=640,bd="2")
frame_right.grid(row=0,column=1)


frame_1=tk.LabelFrame(frame_left,text="Image",width=540,height=500)
frame_1.grid(row=0,column=0)

frame_2=tk.LabelFrame(frame_left,text="Model and Save",width=540,height=140)
frame_2.grid(row=1,column=0)




frame_3=tk.LabelFrame(frame_right,text="Features",width=240,height=640)
frame_3.grid(row=0,column=0)

frame_4=tk.LabelFrame(frame_right,text="Results",width=300,height=640)
frame_4.grid(row=0,column=1,padx=10)



#frame1

def LoadRecord():
    global record_name
    global count
    
    
    count +=1
    if count !=1:
        messagebox(title="Warning",message="Only one image can be opened" )
    else:
        record_name=filedialog.askopenfilename(initialdir="C:\\Users\\OMER\\Desktop\\Dersler",title="load your record")
        print(record_name)
        record_mfcc=python_speech_features_mfcc(record_name)  
        record_mel=Mel_Spectogram(record_name)
        plot_mfcc(record_mfcc,record_name,0)
        plot_mfcc(record_mel,record_name,1)
        
        ##mfcc sonucnun bastırılması
        img=Image.open(record_name+"mfcc.jpg")
        img=imageResize(img)
        img=ImageTk.PhotoImage(img)
        panel=tk.Label(frame_1,image=img)
        panel.image=img
        panel.pack(padx=15,pady=10)
        ## mel sonucunun bastırılması
        img=Image.open(record_name+"mel.jpg")
        img=imageResize(img)
        img=ImageTk.PhotoImage(img)
        panel=tk.Label(frame_1,image=img)
        panel.image=img
        panel.pack(padx=15,pady=10)
        
def imageResize(img):
    basewidth=500
    wprecent=(basewidth/float(img.size[0]))
    hsize=int((float(img.size[1])*float(wprecent)))
    img=img.resize((basewidth,hsize),Image.ANTIALIAS)
    return img

def plot_mfcc(array,adres,info): ## info=0 ise mfcc, info=1 ise mel
    ig, ax = plt.subplots(figsize=(12,5))
    mfcc_data= np.swapaxes(array, 0 ,1)
    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
    #Showing mfcc_data
    plt.tight_layout()
    if info==0:
        pylab.savefig(adres+'mfcc.jpg', bbox_inches=None, pad_inches=0)
    else:
        pylab.savefig(adres+'mel.jpg', bbox_inches=None, pad_inches=0)
    pylab.close()
                         
def python_speech_features_mfcc (adres):
    x, sr = librosa.load(adres,sr=48000)
    y = librosa.effects.trim(x, top_db = 20,frame_length=1200,hop_length=720)[0]
   
    mfcc_speech =python_speech_features.mfcc(signal=y, samplerate=sr, winlen=0.025, winstep=0.01,
                                          numcep=13, nfilt=52, nfft=256, lowfreq=0, highfreq=8000,preemph=0.97,
                                              ceplifter=22,appendEnergy=True)
    
    delta_feat = python_speech_features.delta(mfcc_speech, 2)                           
    
    global record_max_mfcc_delta
    record_max_mfcc_delta=delta_feat.max(0)
    
    global record_max_mfcc 
    record_max_mfcc = mfcc_speech.max(0)
    
    return (mfcc_speech)
    
def Mel_Spectogram(adres):
    y, sr = librosa.load(adres,sr=48000) 
    
   
    mel =( librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                                          fmax=24000) ).T
    global record_max_mel
    record_max_mel=mel.max(0)
    return mel



def save_info():
    first_name_info=firstname.get()
    last_name_info=lastname.get()
    age_info=str(Age.get())
    Sex_info=Sex.get()
    print(first_name_info,last_name_info,age_info,Sex_info)
    file=open("user_info.txt","w")
    file.write("first name of user is: "+first_name_info)
    file.write(" Last name of user is: "+last_name_info)
    file.write(" Age of user is: "+age_info)
    file.write(" Sex of user is: "+Sex_info)
   
    
#####################################################################################
   
menubar=tk.Menu(window)
window.config(menu=menubar)
file=tk.Menu(menubar)
menubar.add_cascade(label="File",menu=file)
file.add_command(label="Open",command=LoadRecord)


######################################################################################
firstname=tk.StringVar()
lastname=tk.StringVar()
Age=tk.IntVar()
Sex=tk.StringVar()

entry_names_for_frame_3=[firstname,lastname,Age,Sex]

####################################################################################

##fram-4

Labels_for_frame_4=["Covid-19","Saglikli"]
for i in range(len(Labels_for_frame_4)):
    x=0.1
    y=(i/10)/2
    tk.Label(frame_4,font=("Times",12),text=str(Labels_for_frame_4[i])+": " ).place(relx=x,rely=y)

# Create an object of tkinter ImageTk
img = ImageTk.PhotoImage(Image.open("C:\\Users\\OMER\\Desktop\\gtu2.jpg"))

# Create a Label Widget to display the text or Image
label = tk.Label(frame_4, image = img).place(relx=0,rely=0.6)

###############################################################
##frame3

Labels_for_frame_3=["First Name","Last Name","Age","Sex"]


for i in range(len(Labels_for_frame_3)):
    x=0.1
    y=(i/10)/2
    tk.Label(frame_3,font=("Times",12),text=str(Labels_for_frame_3[i])+": " ).place(relx=x,rely=y)
    #tk.Entry(frame_3,textvariable=entry_names_for_frame_3[i],width="30").place(relx=x+5,rely=y+5)
    sex_name_entry=tk.Entry(frame_3,textvariable=entry_names_for_frame_3[i],width="10").place(relx=x+0.5,rely=y)
    
register=tk.Button(frame_3,text="Register",width="20",height="2",command=save_info,bg="grey").place(relx=0.1,rely=0.5)

################################################################

#### frame_2

model_selection_label=tk.Label(frame_2,text="Choose your Sex:")
model_selection_label.grid(row=0,column=0,padx=5)

models=tk.StringVar()
model_selection=ttk.Combobox(frame_2,textvariable=models,values=("male","female"),state="readonly")
model_selection.grid(row=0,column=1,padx=5)

##check box
chvar=tk.IntVar()
chvar.set(0)
xbox=tk.Checkbutton(frame_2,text="Save Classification Result",variable=chvar)
xbox.grid(row=1,column=1,pady=5)

##entry
entry=tk.Entry(frame_2,width=23)
entry.insert(string="Saving name...",index=0)
entry.grid()

#################################################################


def classification():
    if record_name != "" and models.get() != "":
       
        
        
        x_test=np.concatenate((record_max_mfcc, record_max_mel,record_max_mfcc_delta), axis=0)
        if models.get()== "male": ## erkek sınıflandırılması yapılacak
            model_filename = "My_voting_model_male.sav"
            my_svm_model = pickle.load(open(model_filename, 'rb'))
            print("modeli yukledik")
            ## x_test[0].reshape(1, -1)
            y_pred = my_svm_model.predict_proba(x_test.reshape(1, -1))
            
            
        else:
            model_filename = "My_voting_model_female.sav"
            my_svm_model = pickle.load(open(model_filename, 'rb'))
            print("modeli yukledik")
            ## x_test[0].reshape(1, -1)
            y_pred = my_svm_model.predict_proba(x_test.reshape(1, -1))

        
        for i in range(len(y_pred)):
                x=0.5
                y=(i/10)/2
                if i !=1:
                    print("i=0")
                    tk.Label(frame_4,bg="red",text=str(y_pred[0][0])).place(relx=0.5,rely=0)
                    tk.Label(frame_4,bg="green",text=str(y_pred[0][1])).place(relx=0.5,rely=0.05)
        
        if y_pred[0][0]>y_pred[0][1]:
           
            sonuc="COVID-19"
        else:
            sonuc="Saglikli"
            
            
        if chvar.get()==1:
            
            val=entry.get()
            entry.config(state="disabled")
            path_name=val+".txt"
            save_txt=record_name+"adresli dosyaya sahip kisinin saglik durumu:"+sonuc
            text_file=open(path_name,"w")
            
            text_file.write(save_txt)
            print("girdi")
            text_file.close()
        else:
            print("save is not select")
            

Classify=tk.Button(frame_3,text="Classify",width="20",height="2",command=classification,bg="grey").place(relx=0.1,rely=0.7)






window.mainloop()