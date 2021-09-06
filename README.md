https://www.modelsw.com/Android/BirdingViaMic/BirdingViaMic.php<br>
predict bird species from songs and calls.<br>
Gerard Gilliland MODEL Software, Inc.<br>
This version is BirdingViaMic CNN<br>
Identify bird songs using the Convolutional Neural Network process<br>
gerardg@modelsw.com<br>
<br>
based on speaker_id.py<br>
performs speaker_id experiments with SincNet.<br>
Mirco Ravanelli<br>
Mila - University of Montreal<br>
https://github.com/mravanelli/SincNet<br>
July 2018<br>
<br>
populated using bird songs from https://www.xeno-canto.org<br>
<br>
build database in SQLite from IOC World Bird List by Frank Gill.<br>
https://www.worldbirdnames.org/new/<br>
<br>
using Convolutional Neural Network model built on Ubuntu 20, Python 3.6, Nvidia Quadro GV100 GPU<br>
run using terminal$ python3 species_id_prediction.py using the GPU<br>
For Android converted to use only CPU.<br>
<br>
build in Android Studio using chaquopy interface to Android<br>
https://chaquo.com/chaquopy/doc/current/examples.html<br>
https://www.youtube.com/watch?v=dFtxLCSu3wQ<br>
run on Android smart phone.<br>
<br>
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.<br>
<br>
this app BirdingViaMicCNN (Convolutional Neural Network) is currently standalone<br>
I plan to integrate it into BirdingViaMic -- but I am not there yet.<br>
this runs but can only handle short songs.<br>
