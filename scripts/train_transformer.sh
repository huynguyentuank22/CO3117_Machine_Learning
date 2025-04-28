export PYTHONPATH=$(pwd)
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
export HYDRA_FULL_ERROR=1

python src/main.py \
    model=transformer \
    trainer.n_epochs=30 \
    data.batch_size=64 \
    optim.lr=0.00002  \