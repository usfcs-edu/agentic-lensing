# put this in your home folder alongside environment.yaml and it should set everything up!
# super simple, this is just very slightly more convenient than a google doc since it get shared alongside the environment.yaml file
# this should be safe to run multiple times, the commands have safeties built in, and their default behavior is to just delete and reinitialize 

CONDA_NAME = "gigalens_modern_env"
KERNEL_NAME = "Gigalens 2.1 Environment"
ENVIRONMENT_FILE = "./environment.yaml"

conda env create -f $ENVIRONMENT_FILE -n $CONDA_NAME
echo "created conda environment"
python -m ipykernel install --user --name $CONDA_NAME --display-name $KERNEL_NAME
echo "created jupyter kernel"

#And that's it! It doesn't have its own version of CUDA, but ideally we should be using the NERSC default