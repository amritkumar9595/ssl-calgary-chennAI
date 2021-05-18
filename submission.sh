
TEST_PATH='/media/student1/RemovableVolume/calgary_new/singlechannel/Test'
TARGET_PATH='/media/student1/RemovableVolume/calgary_new/singlechannel/Test'

BATCH_SIZE=1
DEVICE='cuda:0'


<<ZERO-FILLED
N0_OF_VOLUMES=5
BATCH_SIZE=1
DEVICE='cuda:0'
ACC_FACTOR=4
OUT_DIR='/media/student1/RemovableVolume/singlechannel_calgary_submission/acc_'${ACC_FACTOR}'x/zero_filled/'${N0_OF_VOLUMES}'_volumes'
python submission_zf.py --batch-size ${BATCH_SIZE}  --device ${DEVICE} --out-dir ${OUT_DIR} --test-path ${TEST_PATH}  --acceleration-factor ${ACC_FACTOR} 
ZERO-FILLED

<<ZERO-FILLED
PRETEXT='zero_filled'     # zero_filled  ,scratch , finetuning
VOLUMES=5
ACC_FACTOR=4
PREDICTIONS_PATH='/media/student1/RemovableVolume/singlechannel_calgary_submission/acc_'${ACC_FACTOR}'x/'${PRETEXT}'/'${VOLUMES}'_volumes'
REPORT_PATH='/media/student1/RemovableVolume/reports/acc'${ACC_FACTOR}'x/'${PRETEXT}'/'${VOLUMES}'_volumes'
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
ZERO-FILLED






## scratch training  ##

#<<SUBMISSION
PRETEXT='scratch'
# PRETEXT='finetuning' 
MODEL='dualencoder'
N0_OF_VOLUMES=5
ACC_FACTOR=8
OUT_DIR='/media/student1/RemovableVolume/singlechannel_calgary_submission/acc_'${ACC_FACTOR}'x/'${MODEL}'/'${PRETEXT}'/'${N0_OF_VOLUMES}'_volumes'
MODEL_PATH='/media/student1/NewVolume/MR_Reconstruction/experiments/singlechannel_calgary/acc_'${ACC_FACTOR}'x/'${MODEL}'/'${PRETEXT}'/'${N0_OF_VOLUMES}'_volumes/best_model.pt'
python submission.py --batch-size ${BATCH_SIZE}  --device ${DEVICE} --out-dir ${OUT_DIR} --test-path ${TEST_PATH}  --acceleration-factor ${ACC_FACTOR} --model-path ${MODEL_PATH} --model ${MODEL}
#SUBMISSION

# <<EVALUATE
PREDICTIONS_PATH='/media/student1/RemovableVolume/singlechannel_calgary_submission/acc_'${ACC_FACTOR}'x/'${MODEL}'/'${PRETEXT}'/'${N0_OF_VOLUMES}'_volumes'
REPORT_PATH='/media/student1/RemovableVolume/reports/acc'${ACC_FACTOR}'x/'${MODEL}'/'${PRETEXT}'/'${N0_OF_VOLUMES}'_volumes'
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
# EVALUATE


