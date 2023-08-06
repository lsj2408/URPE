## Code for URPE-based Transformers

### Installation

```shell
pip install -e .
pip install omegaconf==2.0.6
pip install hydra-core==1.0.7
```

### Examples

```shell
# For noPE
bash simulate_noPE.sh

# For RPE
bash simulate_RPE.sh

# For URPE
bash simulate_URPE.sh

```

### Note for Arguments
```shell
# remove absolute positional encodings
--no-use-position-embeddings

# use RPE-based Attention
--rel-pos
--rel-pos-unique # do not share parameters between positions

# use URPE-based Attention
--ur-attn
--ur-attn-unique
--constant-init-ur-attn # initialize the matrix C in URPE to be all-one matrix

# hyperparameters
--vocab-size # vocabulary size, selected from [1000, 10000]
--max-seq-len # sequence length, set as 128

# Tasks:
--task-type 'f' # Even Token Prediction
--task-type 'r' # Position Identification
```