https://www.modelsw.com/Android/BirdingViaMic/BirdingViaMic.php
predict bird species from songs and calls.
Gerard Gilliland MODEL Software, Inc.
This version is BirdingViaMic CNN
Identify bird songs using the Convolutional Neural Network process
gerardg@modelsw.com

based on speaker_id.py
performs speaker_id experiments with SincNet.
Mirco Ravanelli
Mila - University of Montreal
https://github.com/mravanelli/SincNet
July 2018

populated using bird songs from https://www.xeno-canto.org

build database in SQLite from IOC World Bird List by Frank Gill.
https://www.worldbirdnames.org/new/

using Convolutional Neural Network model built on Ubuntu 20, Python 3.6, Nvidia Quadro GV100 GPU
run using terminal$ python3 species_id_prediction.py using the GPU
For Android converted to use only CPU.

build in Android Studio using chaquopy interface to Android
https://chaquo.com/chaquopy/doc/current/examples.html
https://www.youtube.com/watch?v=dFtxLCSu3wQ
run on Android smart phone.

This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

this app BirdingViaMicCNN (Convolutional Neural Network) is currently standalone
I plan to integrate it into BirdingViaMic -- but I am not there yet.
this runs but can only handle short songs.
