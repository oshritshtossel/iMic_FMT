# Recipient independent high accuracy FMT prediction and optimization
This code is attached to the paper "Recipient independent high accuracy FMT prediction and optimization". 
We propose a tool to predict the recipient phenotype a week after the FMT using only the donors' microbiome (iMic).
We then extend the method to optimize the best-planned transplant (Bacterial cocktails) by combining the predictor and a Genetic Algorithm (GA).

## iMic - prediction
iMic predicts a recipient's phenotype t days after the FMT, where t is a parameter of iMic. iMic is based on the donor's properties only. For more details see the paper.

### How to apply iMic

1. Add the dataset to the nni_data_loader function. We recommend to preprocess the data via the MIPMLP pipeline with the following parameters
    Taxonomy Level = Specie, Taxnomy Group = Mean, Normalization = Log. The MIPMLP pipeline can be found at [MIPMLP website](https://mip-mlp.math.biu.ac.il/Home). 
    
2. The rest parameters for the function load_nni_data are:
    name_of_dataset = 'mean', and D_mode = 'dendogram'.
    
3. Use the micro2matrix code to generate the images from the preprocessed microbiome.

4. Use main_nni_runner to apply iMic on the images, the tag can be any recipient's phenotype (such as: Shannon alpha diversity, orders relative abundances, etc. 
    Just change the a_div variable to the target outcome.
    The parmaters that should be used are:
    
    sys[0] = mean (name_of_dataset)
    
    sys[1] = dendogram (D_mode)
    
    sys[2] = cnn2 (model)
    
    #### Notice - iMic can also be appied  separately via the [following pypi](https://pypi.org/project/MIPMLP/)
    

## Genetic Algorithm (GA) - optimization
To find the optimal required FMT given a targeted outcome, such as: maximizing the recipients' Shannon index, minimizing the recipients' Shannon index, or maximizing the relative abundances of a certain order in the recipients' samples, we developed a Genetic Algorithm (GA). For more details see the paper.
    
### How to apply Genetic Algorithm

All the relevant code for the Shannon index task can be found at Genetic_algorithm.py.

1. Prepare a pre-trained iMic model that would be as the oracale of the GA.

2. Define the fitness function, in our case, there is a fitness function of maximizimg or minimizing the recipient's Shannon index a week after FMT while controlling       the number of non-zero taxa in the FMT (gamma). If gamma is 0, it means all the taxa can be used.



## General tools

There is a function for merging different datasets at tools.py.

The Shannon index can be calculated at [this colab](https://colab.research.google.com/drive/173-f8rWrk6lSEkY2dA3Q5HzwqHL9LFeL).


## Running a trained version of iMic
Note there is an option to run a trained version of iMic on the Shannon of the human to human cohort. By getting the weights from [this Drive](https://drive.google.com/file/d/1FIDy8uUBdv9Alj-xTe9Brkl5_QGBwamc/view?usp=sharing) and uploading the files of the mapping, the image of donors, the donors' Shannon and outcome data. The code for applying the model can be found at "run_trained_model.py".


# Contact us

Oshrit Shtossel - oshritvig@gmail.com


    
    
    
    



