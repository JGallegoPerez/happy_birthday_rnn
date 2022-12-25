# happy_birthday_rnn
## Simple Elman network that learns and plays the Happy Birthday song

I hope the script is self-explanatory, I included extensive comments. An Elman network is modeled with two neurons as input and output. The "Happy Birthday" melody is taught to the network, whereby one of the input neurons receives the pitch (from the frequency in Hertz) of each tone that composes the melody, and the other neuron receives the duration (time) of the corresponding tones. I had the choice to input and output the tones as discrete values (for instance, receive an A# as input and expel a C), or to represent them "analogically" across the full frequency spectrum. I thought the latter could offer richer insights, so I went for that approach. It reminds me of the learning of violin and other string instruments that don't have keys or frets to execute the exact same notes. I wonder how it would have all worked out if the data had been represented in a discrete format. 

## Data transformations

The value ranges of the tone frequencies (e.g. an A equals 440Hz) and the durations (e.g. 1 second) are very different. I transformed them in a casual way to place them on similar scales, for both cases always between 0 and 1: I simply divided the frequencies by 1000, and the durations by 10. As a result, they had surprisingly similar scales, while preserving some of their internal properties (for instance, the relationship between notes and their corresponding frequencies is logarithmic). 

## Some results

If I understood the model correctly, it seems that the network can predict the melody quite well. The three things that I can vary in the model are the network size, the number of epochs and the learning rate (lr). While the three need to be tweaked to yield a good fit, I found the learning rate to be perhaps the most important. The default value I found from other tutorials (thus, other tasks) was 0.1. This value was too high for my task. Values greater than 0.03 made the error sometimes increase back too much as the network evolved through the epochs. I found that one "sweet spot" could be around 0.02. Below are two examples where the number of epochs and learning rate were kept constant, with only the network size varying. 

**A not so well predicted input.** However, the size of the network was minimal. One neuron encodes the note pitch (frequency in Hz, graph above) and the other encodes the duration in seconds (graph below):
*input_size, hidden_size, output_size = 7, 5, 2, epochs = 2000, lr = 0.02*

![image](https://user-images.githubusercontent.com/89183135/209466379-c7c3bf15-2f68-48c8-942e-0604c8b157cf.png)
![image](https://user-images.githubusercontent.com/89183135/209466388-3ac04359-6ee6-4f16-a796-3411a4c56778.png)

**The next trained network does a much better prediction.** In fact, a few more thousand epochs would yield a virtually perfect prediction (see last example, after this one).  
*input_size, hidden_size, output_size = 27, 25, 2, epochs = 2000, lr = 0.02*

![image](https://user-images.githubusercontent.com/89183135/209466407-82d240c9-d4ea-456e-9532-5d80d7e503cd.png)
![image](https://user-images.githubusercontent.com/89183135/209466413-c8fb719f-abb2-47aa-9871-db32a1d970d9.png)

**An almost perfect prediction:**
*input_size, hidden_size, output_size = 27, 25, 2, epochs = 5000, lr = 0.02*

![image](https://user-images.githubusercontent.com/89183135/209466431-2fa43afa-dd82-4ce1-b2f5-6b60b23bc18c.png)
![image](https://user-images.githubusercontent.com/89183135/209466436-b1582b52-c2b5-4833-ab22-edc5f4d77b55.png)

## Instructions

I hope it is self-explanatory from the comments in the script. In any case, the parameters we can vary are quite at the top of the code. At the bottom of the code, the actual Happy Birthday melody is generated as audio, followed immediately, for comparison, by the predicted melody. If you intend to run the program many times, I recommend to comment or delete the line that plays the actual Happy Birthday. 

## Dependencies 

torch

numpy

pylab 

pyaudio








