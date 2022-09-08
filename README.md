# ert_cpp
C++ implementation of One Millisecond Face Alignment with an Ensemble of Regression Trees

This is an implementation of the face alignment method by wangefan in 2021.

**About the model**

I can not updata our trained-well model here since the cloud storage limit. 
If needed, please contact me with E-mail: wangefan@gmail.com

**Training data**

I used the lfpw dataset(https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/) 
with 68 landmarks in a face to train our code. 
The training implemetation is in repository https://github.com/wangefan/ert basing on Python.

**Installation**

Clone the repository
build with `./build.sh` to get exe `TestVideoMain.cpp`
put the well train model `ert_model_good.json` (contact me)
run

**Demo**

https://www.youtube.com/watch?v=TUyAcMSuNn0&feature=youtu.be

![image](https://user-images.githubusercontent.com/11495311/189008050-82aabbba-6e52-43a1-b192-691434a3a78e.png)
