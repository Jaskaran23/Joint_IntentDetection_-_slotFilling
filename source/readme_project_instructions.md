### Running evaluation script 

unzip the source folder and run the following command:

cd source


Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt


#### Final submission where f1 score of slot filling is greater than the baseline 

run this command:

a) to check accuracy and f1 on test dataset
 python3 eval.py --intent_true '../output/test_true/trueintent.txt' --slot_true '../output/test_true/trueslots.txt' --intent_predict '../output/test-final/predictedintent-final.txt' --slot_predict '../output/test-final/predictedslots-final.txt'

b) to check accuracy and f1 on validation set

python3 eval.py --intent_true '../output/dev_true/trueintent.txt' --slot_true '../output/dev_true/trueslots.txt' --intent_predict '../output/val-final/predictedintent-final.txt' --slot_predict '../output/val-final/predictedslots-final.txt'





for all other caes: following commands can be used:


#### 1) LSTM with one layer
python3 eval.py --intent_true '../output/dev_true/trueintent.txt' --slot_true '../output/dev_true/trueslots.txt' --intent_predict '../output/lstm1/predictedintent.txt' --slot_predict '../output/lstm1/predictedslots.txt'

#### 2) LSTM with two layers
python3 eval.py --intent_true '../output/dev_true/trueintent.txt' --slot_true '../output/dev_true/trueslots.txt' --intent_predict '../output/lstm2/predictedintent.txt' --slot_predict '../output/lstm2/predictedslots.txt'

#### 3) python3 eval.py --intent_true '../output/dev_true/trueintent.txt' --slot_true '../output/dev_true/trueslots.txt' --intent_predict '../output/gru/predictedintent.txt' --slot_predict '../output/gru/predictedslots.txt'

#### 4) python3 eval.py --intent_true '../output/dev_true/trueintent.txt' --slot_true '../output/dev_true/trueslots.txt' --intent_predict '../output/cases4-8/predictedintent4.txt' --slot_predict '../output/cases4-8/predictedslots4.txt'

#### 5) python3 eval.py --intent_true '../output/dev_true/trueintent.txt' --slot_true '../output/dev_true/trueslots.txt' --intent_predict '../output/cases4-8/predictedintent5.txt' --slot_predict '../output/cases4-8/predictedslots5.txt'



#### 6)python3 eval.py --intent_true '../output/dev_true/trueintent.txt' --slot_true '../output/dev_true/trueslots.txt' --intent_predict '../output/cases4-8/predictedintent6.txt' --slot_predict '../output/cases4-8/predictedslots6.txt'



#### 7) python3 eval.py --intent_true '../output/dev_true/trueintent.txt' --slot_true '../output/dev_true/trueslots.txt' --intent_predict '../output/cases4-8/predictedintent7.txt' --slot_predict '../output/cases4-8/predictedslots7.txt'

#### 8) python3 eval.py --intent_true '../output/dev_true/trueintent.txt' --slot_true '../output/dev_true/trueslots.txt' --intent_predict '../output/cases4-8/predictedintent8.txt' --slot_predict '../output/cases4-8/predictedslots8.txt'


#### 9) python3 eval.py --intent_true '../output/dev_true/trueintent.txt' --slot_true '../output/dev_true/trueslots.txt' --intent_predict '../output/cases9-11/predictedintent9.txt' --slot_predict '../output/cases9-12/predictedslots9.txt'

#### 10)  python3 eval.py --intent_true '../output/dev_true/trueintent.txt' --slot_true '../output/dev_true/trueslots.txt' --intent_predict '../output/cases9-11/predictedintent10.txt' --slot_predict '../output/cases9-12/predictedslots10.txt'


#### 11) python3 eval.py --intent_true '../output/dev_true/trueintent.txt' --slot_true '../output/dev_true/trueslots.txt' --intent_predict '../output/cases9-11/predictedintent11.txt' --slot_predict '../output/cases9-12/predictedslots11.txt'

#### 12) python3 eval.py --intent_true '../output/dev_true/trueintent.txt' --slot_true '../output/dev_true/trueslots.txt' --intent_predict '../output/cases12-15/predictedintent13.txt' --slot_predict '../output/cases12-15/predictedslots13.txt'

#### 13)  python3 eval.py --intent_true '../output/dev_true/trueintent.txt' --slot_true '../output/dev_true/trueslots.txt' --intent_predict '../output/cases12-15/predictedintent14.txt' --slot_predict '../output/cases12-15/predictedslots14.txt'

#### 14) python3 eval.py --intent_true '../output/dev_true/trueintent.txt' --slot_true '../output/dev_true/trueslots.txt' --intent_predict '../output/cases12-15/predictedintent15.txt' --slot_predict '../output/cases12-15/predictedslots15.txt'

#### 15)
case a) testing with dev set:

python3 eval.py --intent_true '../output/dev_true/trueintent.txt' --slot_true '../output/dev_true/trueslots.txt' --intent_predict '../output/cases12-15/valset/predictedintent17.txt' --slot_predict '../output/cases12-15/valset/predictedslots17.txt'


case b) testing with test set 

python3 eval.py --intent_true '../output/test_true/trueintent.txt' --slot_true '../output/test_true/trueslots.txt' --intent_predict '../output/cases12-15/testset/predictedintent17.txt' --slot_predict '../output/cases12-15/testset/predictedslots17.txt'

