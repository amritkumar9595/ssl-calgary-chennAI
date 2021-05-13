MODEL='zf_calgary'   #  dautomap_calgary ,  unet_calgary ,  dualencoder_calgary , zf_calgary
DATASET_TYPE='calgary'
TARGET_PATH='/media/student1/RemovableVolume/calgary_new/singlechannel/Test'



<<ZERO-FILLED
PRETEXT='zero_filled'     # zero_filled  ,scratch , finetuning
VOLUMES=5
ACC_FACTOR=4
PREDICTIONS_PATH='/media/student1/RemovableVolume/singlechannel_calgary_submission/acc_'${ACC_FACTOR}'x/'${PRETEXT}'/'${VOLUMES}'_volumes'
REPORT_PATH='/media/student1/RemovableVolume/reports/acc'${ACC_FACTOR}'x/'${PRETEXT}'/'${VOLUMES}'_volumes'
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
ZERO-FILLED





# <<EVALUATE
PRETEXT='scratch'
# PRETEXT='finetuning'     # zero_filled  ,scratch , finetuning
MODEL='dualencoder'
VOLUMES=1
ACC_FACTOR=8
PREDICTIONS_PATH='/media/student1/RemovableVolume/singlechannel_calgary_submission/acc_'${ACC_FACTOR}'x/'${MODEL}'/'${PRETEXT}'/'${VOLUMES}'_volumes'
REPORT_PATH='/media/student1/RemovableVolume/reports/acc'${ACC_FACTOR}'x/'${MODEL}'/'${PRETEXT}'/'${VOLUMES}'_volumes'
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
# EVALUATE