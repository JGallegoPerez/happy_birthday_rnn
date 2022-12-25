#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import numpy as np
import pylab as pl
import torch.nn.init as init
import time
import pyaudio

dtype = torch.DoubleTensor
input_size, hidden_size, output_size = 27, 25, 2 #Input_size equals hidden_size + output_size
epochs = 500
lr = 0.02
interval = 10 #Interval of epochs for visualization of the error

"""
------------------ABOUT THE DATA---------------------

Frequencies in Hz that correspond to the 12 musical notes (13, including high C):
(C to B corresponds to Do, Re, Mi, Fa, Sol, La, Si)
    
C        261.63
C#/Db    277.18
D        293.66
D#/Eb    311.13
E        329.63
F        349.23
F#/Gb    369.99
G        392.00
G#/Ab    415.30
A        440.00
A#/Bb    466.16
B        493.88
C (high) 523.25

Some rhythmic values (durations) of notes and their relations:

Half note
Quarter note: half the duration of the half note
Eighth note: half the duration of the quarter note

Happy Birthday song in musical notation:
https://en.wikipedia.org/wiki/Happy_Birthday_to_You

"""

#My adaptation of Happy Birthday song into a Tensor.
#First row represents the notes, in order.
#Second row represents the corresponding rhythmic values (durations).
#(half notes, quarter notes and eighth notes are assigned 1, 0.5 and 0.25, respectively).
#Frequencies have been divided by 1000, durations divided by 10. 
#The first pair of values are dummies.
#(Notes in the first row: ccdcfe ccdcgf ccCafed a#a#afgf) 
hb = torch.tensor([[0.26163, 0.26163, 0.29366, 0.26163, 0.34923, 0.32963,
                    0.26163, 0.26163, 0.29366, 0.26163, 0.39200, 0.34923,
                    0.26163, 0.26163, 0.52325, 0.44000, 0.34923, 0.32963, 0.29366,
                    0.46616, 0.46616, 0.44000, 0.34923, 0.39200, 0.34923], 
                   [0.10, 0.10, 0.20, 0.20, 0.20, 0.40,
                    0.10, 0.10, 0.20, 0.20, 0.20, 0.40,
                    0.10, 0.10, 0.20, 0.20, 0.20, 0.20, 0.40,
                    0.10, 0.10, 0.20, 0.20, 0.20, 0.40]])

#For Input: from hb, dummy added as first element. 
x = torch.tensor([[0.26163, 0.26163, 0.26163, 0.29366, 0.26163, 0.34923, 0.32963,
                   0.26163, 0.26163, 0.29366, 0.26163, 0.39200, 0.34923,
                   0.26163, 0.26163, 0.52325, 0.44000, 0.34923, 0.32963, 0.29366,
                   0.46616, 0.46616, 0.44000, 0.34923, 0.39200, 0.34923], 
                  [0.10, 0.10, 0.10, 0.20, 0.20, 0.20, 0.40,
                   0.10, 0.10, 0.20, 0.20, 0.20, 0.40,
                   0.10, 0.10, 0.20, 0.20, 0.20, 0.20, 0.40,
                   0.10, 0.10, 0.20, 0.20, 0.20, 0.40]])



time_steps = np.linspace(1, len(hb[0])+1, len(hb[0])+1)

#Read and Parse data into x, the input
x = torch.transpose(x, 0, 1)     #Input layer
#print(hb)


#Audio player. Takes the (scaled) tensor with tones (Hz) and some values for durations (1,0.5,0.25)
#Adapted from https://stackoverflow.com/questions/8299303/generating-sine-wave-sound-in-python
def audio(tens):
   
    volume = 0.5     # range [0.0, 1.0]
    fs = 44100       # sampling rate, Hz, must be integer
    duration = 1   # in seconds, may be float
    f = 440.0        # sine frequency, Hz, may be float    
    
    for i in range(len(hb[0])): #(Last item will be excluded in the predicted sequence)     
        f = float(tens[0,[i]]*1000) #Obtains the frequency
        duration = float(tens[1,[i]]*10) #Obtains the duration    
        p = pyaudio.PyAudio()
        # generate samples, note conversion to float32 array
        samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)
        # for paFloat32 sample values must be in range [-1.0, 1.0]
        stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fs,
                output=True)
        # play. May repeat with different volume values (if done interactively) 
        stream.write(volume*samples)
        stream.stop_stream()
        stream.close()
        p.terminate()       
        time.sleep(0.2)
    
    time.sleep(2)
 
    
#Build output matrix for prediction    
ya = np.roll(x,-1,axis=0) #Output layer (targets)
y = torch.from_numpy(ya)

mypred = torch.DoubleTensor(len(ya), output_size).type(dtype)

w1 = torch.DoubleTensor(input_size, hidden_size).type(dtype)
init.normal(w1, 0.0, 0.4)
w1 =  Variable(w1, requires_grad=True)
w2 = torch.DoubleTensor(hidden_size, output_size).type(dtype)
init.normal(w2, 0.0, 0.3)
w2 = Variable(w2, requires_grad=True)

def forward(input, context_state, w1, w2):
  xh = torch.cat((input, context_state), 1)
  context_state = torch.tanh(xh.mm(w1))
  out = context_state.mm(w2)
  return  (out, context_state)

for i in range(epochs):
  total_loss = 0
  context_state = Variable(torch.zeros((1, hidden_size)).type(dtype), requires_grad=True)
  for j in range(x.size(0)):
    input = x[j:(j+1)]
#    print("input is " , input)
    target = y[j:(j+1)]
#    print("target is ", target)
    (pred, context_state) = forward(input, context_state, w1, w2)
    mypred[j] = pred
    loss = (pred - target).pow(2).sum()/2
    total_loss += loss
    loss.backward()
    w1.data -= lr * w1.grad.data
    w2.data -= lr * w2.grad.data
    w1.grad.data.zero_()
    w2.grad.data.zero_()
    context_state = Variable(context_state.data)
    if i % interval == 0:
        if j == x.size(0)-1:       
            print("Epoch: {} error {}".format(i, total_loss))
     
context_state = Variable(torch.zeros((1, hidden_size)).type(dtype), requires_grad=False)
prediction = []

prediction = mypred.detach().numpy()

ya_transposed = np.transpose(ya)



#------------------------DATA VISUALIZATION AND AUDIO----------------------------

#SCATTER PLOTS
pl.scatter(time_steps[:-1], (np.transpose(x)[1]*10)[:-1], s=80, label="Input")
pl.scatter(time_steps[1:], (np.transpose(prediction)[1]*10)[:-1], s=50, label="Predicted")
pl.title("Duration (sec) of notes")
pl.xlabel("Order of notes")
pl.ylabel("Duration (s)")
pl.legend()
pl.show()

pl.scatter(time_steps[:-1], (np.transpose(x)[0]*1000)[:-1], s=80, label="Input")
pl.scatter(time_steps[1:], (np.transpose(prediction)[0]*1000)[:-1], s=50, label="Predicted")
pl.title("Frequency (Hz) of notes)")
pl.xlabel("Order of notes")
pl.ylabel("Frequency (Hz)")
pl.legend()
pl.show()


hb_original = hb
hb_predicted = np.transpose(prediction)

#Audio for the original melody
audio(hb_original)

#Audio for the predicted melody
audio(hb_predicted)








