# Targeted Password Guessing Using k-Nearest Neighbors
We propose KNNGuess/KNN-TPG as a new targeted password guessing model based on old password. We show how to use our trained model (with the trained model weights in ./experiment and the trained KNN-TPG database in ./datastore) to evaluate the strength of a current password based on an old one (see Figure 12 in the paper). Some of the data used to generate the experimental plots is shown in ./exp_ans.

# Experimental Setup:
- ## Microarchitecture
  * CPU Model: Intel Xeon Silver processor
  * OS: Ubuntu 20.04 LTS
  * Linux Kernel: 5.11.0-46-generic
  * GPU: NVIDIA RTX 3090 GPU
  * RAM: 256GB
- ## Environment
  * Python 3.7
  * torch 1.13.0+cu116
  * torchvision 0.14.1+cu116
  * word2keypress 1.0.16
  * faiss-gpu 1.7.3


You can use it to build environment:

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install faiss-cpu -c pytorch
```

# Run

## Evaluating password strength
```
python psm.py --source_password YOUR_OLD_PASSWORD --target_password YOUR_NEW_PASSWORD
```
"YOUR_NEW_PASSWORD" is the password whose strength needs to be evaluated. A detailed description of the KNN-PSM can be found in Appendix H of the paper.

The output is the probability of each character in YOUR_NEW_PASSWORD being predicted by KNNGuess. The greater the probability, the less secure the character is, and the easier it is for the password to be guessed by an attacker.
