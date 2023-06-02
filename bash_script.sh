#!/bin/bash

python trainer.py --name surface_roi_ds_0_lap_0 --w_ds 0 --w_laplacian 0
python trainer.py --name surface_roi_ds_025_lap_0 --w_ds 0.25 --w_laplacian 0
python trainer.py --name surface_roi_ds_1_lap_0 --w_ds 1 --w_laplacian 0
python trainer.py --name surface_roi_ds_1_lap_01 --w_ds 1 --w_laplacian 0.1
python trainer.py --name surface_roi_ds_1_lap_001 --w_ds 1 --w_laplacian 0.01
python trainer.py --name surface_roi_ds_1_lap_0001 --w_ds 1 --w_laplacian 0.001