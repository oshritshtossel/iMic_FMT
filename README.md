# Recipient independent high accuracy FMT prediction and optimization
This code is attached to the paper "Recipient independent high accuracy FMT prediction and optimization". 
We propose a tool to predict the recipient phenotype a week after the FMT using only the donors' microbiome (iMic).
We then extend the method to optimize the best-planned transplant (Bacterial cocktails) by combining the predictor and a genetic algorithm (GA).

## iMic
To apply iMic, we need:

1. To add our dataset to the nni_data_loader function. We recommend to preprocess the data via the MIPMLP pipeline.
    The MIPMLP pipeline can be found at [MIPMLP website](https://mip-mlp.math.biu.ac.il/Home)


