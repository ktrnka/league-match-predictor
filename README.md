# README #

Code for crawling Riot Games API, populating a MongoDB, and running machine learning experiments to predict who will win the games.

## Note
This is a fork from my main repo with the configuration/API keys removed from the history. Unfortunately the data is too big to include in the repository itself.

## Setup

* Install from requirements.txt.
* Small dataset: https://dl.dropboxusercontent.com/u/2094014/league-predictor/matches_10_20_50k.csv.gz
* Medium dataset: https://dl.dropboxusercontent.com/u/2094014/league-predictor/matches_10_13_200k.csv.gz

## Running

From the src directory:
```PYTHONPATH=$PYTHONPATH:. python exploration/train_test.py --xg --n-jobs 3 path/to/dataset```
