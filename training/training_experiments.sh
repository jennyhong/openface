## The trainFirstHalf flag and the -cache flag should always be used together
## When using the trainFullStudent flag, the -firstHalfCached file should be taken from the directory
## of the -cache directory from the last run (the run corresponding to the first half of the full net)

DATA_DIR=../data/lfw/aligned-pruned
MODEL_DIR=../models/fitnets
WORK_DIR=work-firsthalf
FINAL_WORK_DIR=work-wholefitnet
NEPOCHS=3
USE_GPU=false
if $USE_GPU ; then
  NEPOCHS=1000
fi

rm -r $WORK_DIR*
rm -r $FINAL_WORK_DIR*

for ((  i = 1 ;  i <= 1;  i++  ))
do
  if $USE_GPU ; then
    qlua main.lua -data $DATA_DIR -modelDef $MODEL_DIR/fitnets_firsthalf$i.def.lua -cuda -cudnn -trainFirstHalf -cache $WORK_DIR-$i  -nEpochs $NEPOCHS;
    qlua main.lua -data $DATA_DIR -modelDef $MODEL_DIR/fitnets_all$i.def.lua -cuda -cudnn -trainFullStudent -cache $FINAL_WORK_DIR-$i  -firstHalfCached $WORK_DIR-$i/1/model_$NEPOCHS.t7;
  else
    qlua main.lua -data $DATA_DIR -modelDef $MODEL_DIR/fitnets_firsthalf$i.def.lua -trainFirstHalf -cache $WORK_DIR-$i  -nEpochs $NEPOCHS -peoplePerBatch 5 -imagesPerPerson 5 -epochSize 5;
    qlua main.lua -data $DATA_DIR -modelDef $MODEL_DIR/fitnets_all$i.def.lua -trainFullStudent -cache $FINAL_WORK_DIR-$i  -firstHalfCached $WORK_DIR-$i/1/model_$NEPOCHS.t7 -nEpochs $NEPOCHS -peoplePerBatch 5 -imagesPerPerson 5 -epochSize 5;
  fi
done
