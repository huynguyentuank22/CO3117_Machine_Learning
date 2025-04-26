export PYTHONPATH=""
export PYTHONPATH=$(pwd)
echo PYTHONPATH=$PYTHONPATH
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
export HYDRA_FULL_ERROR=1

python src/main.py \
    model=rnn \
    trainer.n_epochs=10 \
    data.batch_size=64 \
    optim.lr=0.002 \