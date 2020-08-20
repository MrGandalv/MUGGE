# Project M.U.G.G.E. - Mostly Unfinished Genre Grading Engine

## Description
We are Students of the University of Hamburg (Germany) and in this project we try to recognize the genre of a given music file. It was important to us that the user has as much freedom as possible and hence the user can design the structure of the analysing algorithm in three steps:

1. select all the features from the feature options (see below) you wish to be taken into account  
2. select an arbitrary amount of base learners (you have to choose a machine learning method for each one)
3. select an ensemble learner from the ensemble learner list (see below)

**feature options:**
- zero crossingrate
- spectral centroid
- mel-frequency cepstral coefficients (mfcc)
- chroma frequencies
- chords
- the 'all' feature (which are all the feature already mentioned)
-'all except cords'
(in future versions the user shall be able to make an own collection of features)

**ensemble learner options:**
- bagging from sklearn
- MLP Neural Network that takes into account all the base learners that have been selected in step 2 (see above)
(in the future we plan to include more ensemble learners for an own handpicked collection of base learners)

### dataset

We used the GTZAN dataset for music classification. It consists out of 10 genres with 100 audio files each, all having a length of 30 seconds.
(we downloaded it from [kaggle](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification?))

In the current version we splitted each music file into 3s long pieces.

### funfact

Mugge (or also "Mucke" or "Mukke") is an informal German synonym for a music performance (and inofficially but widely spread also for music itself).

## Results
**ATTENTION**: *the current code isn't stable. The results refere to the original GTZAN form with 30s long music files*

- **best ensemble learner** was the bagging learner (given by sklearn) with an accuracy of 73.5 % (last stable version):

![](https://github.com/MrGandalv/MUGGE/blob/master/doc/Bagging_Decision.png)

- **best base learner** was the MLP Neural Network (given by sklearn) with an accuracy of 72.1 % (last stable version):

![](https://github.com/MrGandalv/MUGGE/blob/master/src/barchart_accuracy_whole_songs.png)

## To be done in the future

- [ ] build a GUI
- [ ] the user shall be able to make an own collection of features
- [ ] include more ensemble learners for an own handpicked collection of base learners
- [ ] include own base learners like given Neural Networks (CNN's and RNN's) done with tensorflow
