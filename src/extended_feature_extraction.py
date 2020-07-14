import numpy as np
import librosa

import os
import os.path

import csv
import time
import warnings

#  Suppress a weird Future warning, which is (hopefully) not important.
warnings.filterwarnings('ignore')

#  Use the time module to measure how long the program is running.
#  First get the current time.
starttime = time.time()


#  The following function is an auxiliary function for the main function 'write_feature_file'.
#  Here, for the 'mode' parameter only the states "w" (write) and "a" (append) are admissible.
#  If "a" is committed, the function opens the .csv file 'file_name', and 'line_to_add' is added
#  to the end of the file (in a new line). Gives an error, when 'file_name' does not exist.
#  If "w" is committed, it creates an empty .csv file with the name 'file_name', and 'line_to_add' is added
#  in the file (in the first line).
#  Important: In writting mode, an existing file with the same name will be erased/ overwritten.
#  The given file 'file_name' must be a .csv file.
#  The "with" statement ensures that the file will automatically be closed afterwards.


def append_data_to_file(features_file_name, line_to_add, mode):
    assert features_file_name.endswith(".csv"), "The file 'file_name' must be a .csv file."
    assert mode in ["w", "a"], "The 'mode'-parameter does not follow the intended function of this script. "
    file = open(features_file_name, mode, newline="")
    with file:
        writer = csv.writer(file)
        writer.writerow(line_to_add)


#  The 'write_feature_file' is the main function of this script.
#  The functions goal is to create a .csv file named 'file_name', with specified music features of
#  a given dataset of songs in it. Usually, the file can then be found in the directory of your current python
#  project. The created file starts with a headline, consisting of "filename", followed by the names of
#  the extracted features and ending with "label", i.e. the genre of the song. Every other row persists of the
#  extracted data of one song, structured in the same order as the headline.
#
#  The parameters 'file_name' (name of the desired .csv file)
#  and 'dataset_path' (the path, where the dataset is stored) should be committed as strings, e.g.
#  file_name= "music_features.csv",
#  dataset_path = "C:/Users/MaxM/PycharmProjects/music_project/music_data".
#
#  With the list of strings 'genrelist' and 'names' one can specify, which genres from the database
#  should be considered and which features should be extracted. Use the .split() method to easily create
#  the wanted format from a pure string, e.g. genrelist="rock pop".split(),
#  names="chroma_stft spectral_centroid zero_crossing_rate mfcc".split() .
#  Important: Dont forget spaces between the words! Also, right now, only the 4 features
#  "chroma_stft", "spectral_centroid", "zero_crossing_rate" and "mfcc" are supported. If other features are commited,
#  they wont be extracted and they cant be seen in the file. An Assertion-Error gets raised, if 'names'
#  contains no supported features at all. The "mfcc" feature consists of 20 different features.
#
#  The integer 'max_songs_per_genre' can be used to limit the amount of extracted songs per genre,
#  e.g max_songs_per_genre=10 will extract features only from the first 10 songs from every genre.
#  A small number is useful to test, if all parameters are correctly committed and the created file has
#  the wanted format and information.
#
#  The last (string-)parameter 'overwrite' should be used in the following way: Only if explicitly "Yes" is
#  plugged in, an existing file with the same name as 'file_name' can be overwritten/erased. Thus, the
#  goal is here to prevent an accidental loss of an existing file, by setting, e.g. overwrite="".
#  Another option is to commit overwrite="Append". In this case one can extend an existing file with
#  the same name as 'file_name' by more data. This option is only useful to append different genres
#  step by step, not to add more features to an existing file, because then the structure of the file wont
#  be consistent.
#
#  A complete example for the use of the function is given below.
#
def write_feature_file(features_file_name, dataset_path, genrelist, featurelist, max_songs_per_genre, overwrite):
    does_features_file_exist = os.path.isfile(features_file_name)
    does_dataset_path_exist = os.path.isdir(dataset_path)
    assert does_dataset_path_exist, "The  path 'dataset_path' cant be found."
    assert not does_features_file_exist or overwrite in "Yes Append".split(), "Choose a new 'file-name', if you dont want to overwrite it."
    c, s, z, m = 0, 0, 0, 0
    headline = "filename"
    if "chroma_stft" in featurelist:
        c = 1
        headline += " chroma_stft_mean chroma_stft_var"
    if "spectral_centroid" in featurelist:
        s = 1
        headline += " spectral_centroid_mean spectral_centroid_var"
    if "zero_crossing_rate" in featurelist:
        z = 1
        headline += " zero_crossing_rate_mean zero_crossing_rate_var"
    if "mfcc" in featurelist:
        m = 1
        for i in range(1, 21):  # remember: the mfcc feature consists of 20 different features
            headline += f" mfcc{i}_mean mfcc{i}_var"
    headline += " label"
    assert max(c, s, z, m) > 0, "There are no supported features given in 'names'."
    if overwrite == "Append":
        assert does_features_file_exist, "Attention: Append not possible, because 'file_name' cant be found. "
    else:
        append_data_to_file(features_file_name, headline.split(), "w")
    for genre in genrelist:
        i = 0
        print(genre)  # shows the progress of the programm, while running. This line can be deleted, if not wanted.
        for filename in os.listdir(f"{dataset_path}/{genre}"):
            if i >= max_songs_per_genre:
                break
            songname = f"{dataset_path}/{genre}/{filename}"
            y, sr = librosa.load(songname, mono=True,
                                 duration=30)  # choose with duration , how much is loaded (in seconds)
            line_to_add = f"{filename}"
            if c == 1:
                c_s = librosa.feature.chroma_stft(y=y, sr=sr)
                line_to_add += f" {np.mean(c_s)} {np.var(c_s)}"
            if s == 1:
                s_c = librosa.feature.spectral_centroid(y=y, sr=sr)
                line_to_add += f" {np.mean(s_c)} {np.var(s_c)}"
            if z == 1:
                z_c_r = librosa.feature.zero_crossing_rate(y)
                line_to_add += f" {np.mean(z_c_r)} {np.var(z_c_r)}"
            if m == 1:
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                for mfeat in mfcc:
                    line_to_add += f" {np.mean(mfeat)} {np.var(mfeat)}"
            line_to_add += f" {genre}"
            append_data_to_file(features_file_name, line_to_add.split(), "a")
            i += 1


#               -----Example----
if __name__ == '__main__':
    features_file_name = "all_features_whole_songs.csv"
    # dataset_path = "C:/Users/MaxM/PycharmProjects/music_project/data_music"
    genrelist = "rock pop disco blues classical country hiphop jazz metal reggae".split()
    # genrelist = "rock pop disco blues jazz".split()
    featurelist = "chroma_stft spectral_centroid zero_crossing_rate mfcc".split()
    my_dataset_path = "C:/Users/JD/PycharmProjects/newstart/data_music"
    #  Now the following command should create a data file, which consists of a headline and 4 (rsp.23 because of mfcc)
    #  features from 5 songs per genre:
    #
    #write_feature_file(features_file_name, my_dataset_path, genrelist, featurelist, 100, "")
    #
    #  should take around 30 seconds.

    #  Prints out how long the program was running, in seconds.
    endtime = time.time()
    print("{:5.3f}s".format(endtime - starttime))

    #  main source: https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8
