from model import Net

if __name__ == '__main__':
    from pytorch_lightning import Trainer

    model = Net(
        # root='./data',
        root='/mnt/data/datasets/cross-mapping-4s',
        batch_size=1
    )

    # trainer = Trainer(gpus=-1, nb_sanity_val_steps=3)
    trainer = Trainer(gpus=-1, nb_sanity_val_steps=1, min_nb_epochs=500, distributed_backend='ddp', early_stop_callback=False)
    trainer.fit(model)
