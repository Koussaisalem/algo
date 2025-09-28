from models.surrogate import model
import schnetpack as spk
import torch
import torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
import numpy as np
from schnetpack.datasets import QM9
# importi QM9 jdidq feha energy AS QM9 
output_energy = spk.task.ModelOutput(
    name=QM9.energy,
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.01,
    metrics={"MAE": torchmetrics.MeanAbsoluteError()},
)

output_forces = spk.task.ModelOutput(
    name=QM9.forces,
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.99,
    metrics={"MAE": torchmetrics.MeanAbsoluteError()},
)

task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_energy, output_forces],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4},
)


logger = pl.loggers.TensorBoardLogger(save_dir=forcetut)
callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join(forcetut, "best_inference_model"),
        save_top_k=1,
        monitor="val_loss",
    )
]

trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,
    default_root_dir=forcetut,
    max_epochs=5,  # for testing, we restrict the number of epochs
)
trainer.fit(task, datamodule=ethanol_data)