export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
export HYDRA_FULL_ERROR=1

python src/evaluate.py \
    model=transformer