# About

This project aims to create fill in the blanks question from a given textbook. Currently the model works only for biology, chemistry and physics texts, but we aim to expand the project.

We created a training database using some sample texts. We extracted some features such as the position of the sentence in the paragraph, is the sentence the first sentence of the para, number of keywords in the sentence etc. For word selection we used TF and IDF values to select keywords.

# Guide

Download the repo, unrar and open the folder then run the following commands.

pip3 -r requirements.txt

python3 main.py [path to input text file] arg2

Where:
      
      arg2 = c for chemistry text

      arg2 = b for bio text
      
      arg2 = p for physics text
