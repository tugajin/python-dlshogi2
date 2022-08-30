while :
do
  CHECKPOINTS_NAME=`ls -t checkpoints/ | head -n 1`
  TRAIN_FILE=`ls -1 data/ | head -n 1`
  python -m pydlshogi2.train data/$TRAIN_FILE floodgate_teacher-2013-01 -r checkpoints/$CHECKPOINTS_NAME -e 1
  rm -f data/$TRAIN_FILE
done
