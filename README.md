# Music-Genre-Prediction-Using-RNN-LSTM
RNN-LSTM and Deep learning models are build which is used to predict the genre of the song which is given as input to the model. 
<br>
<br>
# Dataset-Processing
Here, the dataset used is GTZAN music dataset found from this link http://marsyas.info/downloads/datasets.html
<br>
The Dataset Preprocessing.py file is used to preprocess the data. It creates a json file with all the lables or categories and features of each song. The features are extracted using the mfcc. Each song is split into 10 segments and then for each segment the mfcc is extracted and stored in the json file. The number of segments is a variable parameter which can be chosen by you. 
The json file is used to train the model.
<br>
# Model-training
The model is trained using tensorflow and keras over it. It is trained using RNN-LSTM, run for 50 epochs. It gave a train accuracy of 94% and test accuracy of 75%.
<br>
# Model-prediction
Prediction of genre of a song is done by splitting the entire song into 30 second clips and each of the 30 second clips are split into 5 segments and then prediction is done on each segment. The most predicted genre out of all predictions is taken as the overall predicition. This is done as the model is train with 3 seconds audio file snippets.
 
# Requirements 
For this project, you would require to have the following packages - 
<ul>
  <li> Tensorflow </li?
  <li> Keras </li>
  <li> librosa </li>
  <li> Pydub </li>
  <li> json </li>
  <li> numpy </li>
  <li> Matplotlib </li>
</ul>
Apart from this, you would be required to have ffmpeg installed in your system and added to the path. This is for mp3 to wav conversion code.
