export PYTHONPATH=""
export PYTHONPATH=$(pwd)
echo PYTHONPATH=$PYTHONPATH
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
export HYDRA_FULL_ERROR=1

python src/main.py \
    model=lstm \
    trainer.n_epochs=3 \
    data.batch_size=64 \