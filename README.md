# DCGAN-huggingface

An implementation of DCGAN, leveraging the HuggingFace ecosystem for getting data and pushing to the hub.

To train the model with the default parameters on MNIST, simply do:

```
python train.py
```

This will train the model for 5 epochs. You can specify another dataset from the [HuggingFace hub](https://huggingface.co/) as follows:

```
python train.py --dataset cifar-10
```

## Training on your own data

You can of course also train on your own images. For this, one can leverage Datasets' [ImageFolder](https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder). Make sure to authenticate with the hub first, by running the `huggingface-cli login` command in a terminal, or the following in case you're working in a notebook:

```
from huggingface_hub import notebook_login

notebook_login()
```

Next, run the following in a notebook/script:

```
from datasets import load_dataset

# first: load dataset
# option 1: from local folder
dataset = load_dataset("imagefolder", data_dir="path_to_folder")
# option 2: from remote URL (e.g. a zip file)
dataset = load_dataset("imagefolder", data_files="URL")

# next: push to the hub
dataset.push_to_hub("huggan/my-awesome-dataset")
```

You can then simply pass the name of the dataset to the script:

```
python train.py --dataset huggan/my-awesome-dataset
```

## Pushing model to the hub

You can push your trained generator to the hub after training by specifying the `push_to_hub` flag. 
Then, you can run the script as follows:

```
python train.py --push_to_hub --model_name dcgan-mnist
```
