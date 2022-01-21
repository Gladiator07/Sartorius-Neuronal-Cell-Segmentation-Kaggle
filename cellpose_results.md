# Experiments to perform

## Cellpose

| Description | fold | cv | Public LB |
| ----------- | ----- | --- | --- |
| resample false, 500 epochs, SGD, cyto2 | 0 | 0.287 | 0.310
|                                        | 1 | 0.2809 | 0.306
|                                        | 2 | 0.2925 | 0.309
|                                        | 3 | 0.2725 | 0.310
|                                        | 4 | 0.2586 | 0.305
| -- | -- | -- | -- | --|
| resample false, 1000 epochs, SGD, cyto2 | 0 | 0.2901 | 0.311
|                                        | 1 | 0.295 | 0.314
|                                        | 2 | 0.2942 | 0.311
|                                        | 3 | 0.2834 | 0.315
|                                        | 4 |  0.2600 | 0.308