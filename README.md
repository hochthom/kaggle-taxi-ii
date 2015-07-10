kaggle-taxi-trip-time-prediction
================================

Winning solution for the Taxi-Trip Time Prediction Challenge on Kaggle (https://www.kaggle.com/c/pkdd-15-taxi-trip-time-prediction-ii).

Documentation about the method is available in `doc/Taxi_II_Winning_Solution.pdf`. 
Information on how to generate the solution file can be found below.

## Generating the solution

### Install the dependencies

I used [python]() 2.7.9 with its popular scientific modules, in particular 
- [numpy](http://www.numpy.org/) for feature processing,
- [pandas](http://pandas.pydata.org/) for data handling, and 
- [sklearn](http://scikit-learn.org/stable/) for model training.

### Download the code

To download the code, run:

```
git clone git://github.com/hochthom/kaggle-taxi-ii.git
```

### Download the training data

Download the data files from [Kaggle](https://www.kaggle.com/c/pkdd-15-taxi-trip-time-prediction-ii/data). Place and extract the files in the data directory.

### Create the training and test sets

Run the following scripts in any order.

```
create_training_set_N1.py
```
Result: train_pp_N1.csv with size: 1658559 x 8, Duration:  679.9 sec *

```
create_training_set_N2.py
```
Result: train_pp_N2.csv with size: 1658559 x 10, Duration: 771.8 sec *

```
create_training_set_N3.py
```
Result: train_pp_N3.csv with size: 1658559 x 10, Duration: 819.9 sec *

```
create_training_set.py
```
Result: train_pp_RND.csv with size: 1928087 x 17, Duration: 911.5 sec *

```
create_training_set_Experts.py
```
Result: train_pp_TST_0.csv - train_pp_TST_319.csv, no file for trip 125
319 Training Sets, size 3 - 109915 x 17, Duration: 10135.2 sec *

*Note: The listed running times were achieved on a 12-core server with 24GByte of ram. 
The running time may be considerably longer if you use a machine with less ram!

### Create the single model predictions

```
mk_submission.py
```
creates submissions for models N1, N2, N3, RND; Duration: 5175.5 sec *
Result: my_submission_xx.csv with xx in [N1,N2,N3,RND]

```
mk_submission_Experts.py
```
creates the expert model predictions, Duration: 2968.3 sec *
Result: predictions_TVT_experts.pkl

*Note: The listed running times were achieved on a 12-core server with 24GByte of ram. 
The running time may be considerably longer if you use a machine with less ram!

### Create the final (blended) submissions

To generate blended prediction files from all the models, run:

```
blend_submissions.py
```
Result: final_submission_1.csv and final_submission_2.csv

### Submit predictions

Submit the files on [Kaggle](https://www.kaggle.com/c/pkdd-15-taxi-trip-time-prediction-ii/submissions) to get it scored. 

Submission 1 scored better in the private leaderboard than submission 2 (0.5092 vs. 0.5045), whereas for the public leaderboard it was the other way around (0.5253 vs. 0.5354).

