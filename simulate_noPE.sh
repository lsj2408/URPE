TOTAL_UPDATES=40000
WARMUP_UPDATES=6000
PEAK_LR=7e-5
TOKENS_PER_SAMPLE=128 # Max sequence length
MAX_SENTENCES=32
UPDATE_FREQ=4
VOCAB_SIZE=1000 # [1000, 10000]
DATA_DIR=no_data # online generate sequence data

# Task ETP
task_type='f'
TSB_DIR=../output/task-ETP-noPE/tsb
SAVE_DIR=../output/task-ETP-noPE

python train.py --fp16 $DATA_DIR \
    --task synthetic_task --no-use-position-embeddings \
    --criterion synthetic_task --task-type $task_type \
    --dataset-valid-size 6400 --max-seq-len $TOKENS_PER_SAMPLE --vocab-size $VOCAB_SIZE \
    --arch roberta_base --encoder-layers 3 --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE --required-batch-size-multiple 1 \
    --optimizer adam --adam-betas '(0.9,0.999)' --adam-eps 1e-8 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.0 --attention-dropout 0.0 --weight-decay 0.0 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 20 \
    --tensorboard-logdir $TSB_DIR --save-dir $SAVE_DIR --no-epoch-checkpoints

# Task PI
task_type='r'
TSB_DIR=../output/task-PI-noPE/tsb
SAVE_DIR=../output/task-PI-noPE

python train.py --fp16 $DATA_DIR \
    --task synthetic_task --no-use-position-embeddings \
    --criterion synthetic_task --task-type $task_type \
    --dataset-valid-size 6400 --max-seq-len $TOKENS_PER_SAMPLE --vocab-size $VOCAB_SIZE \
    --arch roberta_base --encoder-layers 3 --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE --required-batch-size-multiple 1 \
    --optimizer adam --adam-betas '(0.9,0.999)' --adam-eps 1e-8 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.0 --attention-dropout 0.0 --weight-decay 0.0 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 20 \
    --tensorboard-logdir $TSB_DIR --save-dir $SAVE_DIR --untie-weights-roberta --no-epoch-checkpoints
