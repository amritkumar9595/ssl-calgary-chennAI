MODEL='unet'
DATASET_TYPE='calgary'
BASE_PATH='/media/student1/NewVolume/MR_Reconstruction'

TRAIN_PATH='/media/student1/RemovableVolume/calgary_new/singlechannel/Train'
VALIDATION_PATH='/media/student1/RemovableVolume/calgary_new/singlechannel/Val'

DEVICE='cuda:0'
BATCH_SIZE=1
NUM_EPOCHS=100
                                          ### U-Net  ###



<<TRAINING_U-NET_FROM_SCRATCH
LEARNING_RATE=0.0001
N0_OF_VOLUMES=5
ACC_FACTOR=8
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/singlechannel_calgary/acc_'${ACC_FACTOR}'x/unet/scratch/'${N0_OF_VOLUMES}'_volumes'
python train_unet_scratch.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}  --sample ${N0_OF_VOLUMES} --lr ${LEARNING_RATE} 
TRAINING_U-NET_FROM_SCRATCH


<<PRETRAINING_U-NET
LEARNING_RATE=0.0001
N0_OF_VOLUMES=25
BATCH_SIZE=1
ACC_FACTOR=8
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/singlechannel_calgary/acc_'${ACC_FACTOR}'x/unet/pretext'
python train_unet_pretext.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}  --sample ${N0_OF_VOLUMES} --lr ${LEARNING_RATE} 
PRETRAINING_U-NET



<<FINETUNING_U-NET
BATCH_SIZE=1
LEARNING_RATE=0.0001
N0_OF_VOLUMES=5
ACC_FACTOR=8
PRETEXT_MODEL='/media/student1/NewVolume/MR_Reconstruction/experiments/singlechannel_calgary/acc_'${ACC_FACTOR}'x/unet/pretext/best_model.pt'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/singlechannel_calgary/acc_'${ACC_FACTOR}'x/unet/finetuning/'${N0_OF_VOLUMES}'_volumes'
python train_unet_finetune.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}  --sample ${N0_OF_VOLUMES} --lr ${LEARNING_RATE} --pretext ${PRETEXT_MODEL} 
FINETUNING_U-NET
      
                                             ### Dual-encoder  ###

<<TRAINING_DUALENCODER_FROM_SCRATCH
LEARNING_RATE=0.00015
N0_OF_VOLUMES=20
ACC_FACTOR=8
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/singlechannel_calgary/acc_'${ACC_FACTOR}'x/dualencoder/scratch/'${N0_OF_VOLUMES}'_volumes/'
python train_dualencoder_scratch.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}  --sample ${N0_OF_VOLUMES} --lr ${LEARNING_RATE} 
TRAINING_DUALENCODER_FROM_SCRATCH



<<PRETRAINING_OF_DUALENCODER
LEARNING_RATE=0.00015
N0_OF_VOLUMES=25
BATCH_SIZE=1
ACC_FACTOR=8
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/singlechannel_calgary/acc_'${ACC_FACTOR}'x/dualencoder/pretext/'
python train_dualencoder_pretext.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}  --sample ${N0_OF_VOLUMES} --lr ${LEARNING_RATE} 
PRETRAINING_OF_DUALENCODER



<<FINETUNING_OF_DUALENCODER
LEARNING_RATE=0.00015
N0_OF_VOLUMES=20
ACC_FACTOR=8
PRETEXT_MODEL='/media/student1/NewVolume/MR_Reconstruction/experiments/singlechannel_calgary/acc_'${ACC_FACTOR}'x/dualencoder/pretext/best_model.pt'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/singlechannel_calgary/acc_'${ACC_FACTOR}'x/dualencoder/finetuning/'${N0_OF_VOLUMES}'_volumes/'
python train_dualencoder_finetune.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}  --sample ${N0_OF_VOLUMES} --lr ${LEARNING_RATE} --pretext ${PRETEXT_MODEL} 
FINETUNING_OF_DUALENCODER



                                             ### W-Net ###


<<TRAINING_WNET_FROM_SCRATCH
BATCH_SIZE=1
LAMBDAA=0.1
LEARNING_RATE=0.001
N0_OF_VOLUMES=25
ACC_FACTOR=4
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/singlechannel_calgary/acc_'${ACC_FACTOR}'x/wnet/scratch/'${N0_OF_VOLUMES}'_volumes'
python train_wnet_scratch.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}  --sample ${N0_OF_VOLUMES} --lr ${LEARNING_RATE} --lambdaa ${LAMBDAA}
TRAINING_WNET_FROM_SCRATCH


<<PRETRAINING_OF_WNET
LAMBDAA=0.1
LEARNING_RATE=0.001
N0_OF_VOLUMES=25
BATCH_SIZE=1
ACC_FACTOR=8
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/singlechannel_calgary/acc_'${ACC_FACTOR}'x/wnet/pretext'
python train_wnet_pretext.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}  --sample ${N0_OF_VOLUMES} --lr ${LEARNING_RATE} --lambdaa ${LAMBDAA}
PRETRAINING_OF_WNET

<<FINETUNING_OF_WNET
LAMBDAA=0.1
LEARNING_RATE=0.001
N0_OF_VOLUMES=25
BATCH_SIZE=1
ACC_FACTOR=4
PRETEXT_MODEL='/media/student1/NewVolume/MR_Reconstruction/experiments/singlechannel_calgary/acc_'${ACC_FACTOR}'x/wnet/pretext/best_model.pt'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/singlechannel_calgary/acc_'${ACC_FACTOR}'x/wnet/finetuning/'${N0_OF_VOLUMES}'_volumes'
python train_wnet_finetune.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}  --sample ${N0_OF_VOLUMES} --lr ${LEARNING_RATE} --pretext ${PRETEXT_MODEL} --lambdaa ${LAMBDAA}
FINETUNING_OF_WNET


                                             ### dAutomap ###



# <<TRAINING_dAUTOMAP_FROM_SCRATCH
BATCH_SIZE=1
LEARNING_RATE=0.0001
N0_OF_VOLUMES=5
ACC_FACTOR=8
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/singlechannel_calgary/acc_'${ACC_FACTOR}'x/dautomap/scratch/'${N0_OF_VOLUMES}'_volumes'
python train_dautomap_scratch.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}  --sample ${N0_OF_VOLUMES} --lr ${LEARNING_RATE} 
# TRAINING_dAUTOMAP_FROM_SCRATCH


<<PRETRAINING_OF_dAUTOMAP
BATCH_SIZE=1
LEARNING_RATE=0.0001
N0_OF_VOLUMES=25
BATCH_SIZE=1
ACC_FACTOR=8
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/singlechannel_calgary/acc_'${ACC_FACTOR}'x/dautomap/pretext'
python train_dautomap_pretext.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}  --sample ${N0_OF_VOLUMES} --lr ${LEARNING_RATE} 
PRETRAINING_OF_dAUTOMAP



# <<FINETUNING_OF_WNET
BATCH_SIZE=1
LEARNING_RATE=0.0001
N0_OF_VOLUMES=5
BATCH_SIZE=1
ACC_FACTOR=8
PRETEXT_MODEL='/media/student1/NewVolume/MR_Reconstruction/experiments/singlechannel_calgary/acc_'${ACC_FACTOR}'x/dautomap/pretext/best_model.pt'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/singlechannel_calgary/acc_'${ACC_FACTOR}'x/dautomap/finetuning/'${N0_OF_VOLUMES}'_volumes'
python train_dautomap_finetune.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}  --sample ${N0_OF_VOLUMES} --lr ${LEARNING_RATE} --pretext ${PRETEXT_MODEL}
# FINETUNING_OF_WNET