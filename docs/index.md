# SNRAware

This repository contains the Pytorch code in our paper [SNRAware: Improved Deep Learning MRI Denoising with Signal-to-noise Ratio Unit Training and G-factor Map Augmentation](https://pubs.rsna.org/doi/full/10.1148/ryai.250227) published at the Radiology: Artificial Intelligence:

```latex
@article{
    doi:10.1148/ryai.250227,
    author = {Xue, Hui and Hooper, Sarah M. and Pierce, Iain and Davies, Rhodri H. and Stairs, John and Naegele, Joseph and Campbell-Washburn, Adrienne E. and Manisty, Charlotte and Moon, James C. and Treibel, Thomas A. and Hansen, Michael S. and Kellman, Peter},
    title = {SNRAware: Improved Deep Learning MRI Denoising with Signal-to-noise Ratio Unit Training and G-factor Map Augmentation},
    journal = {Radiology: Artificial Intelligence},
    volume = {7},
    number = {6},
    pages = {e250227},
    year = {2025},
    doi = {10.1148/ryai.250227},
    note ={PMID: 41123451},
    URL = {https://doi.org/10.1148/ryai.250227}
}
```

- Model type: Imaging AI, non-generative
- License: MIT

## Get started

`uv` is used in this project. Please install it as:

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# install git-lfs
sudo apt update
sudo apt install git-lfs direnv
```

Make sure commands `uv` are on your path.

Also, this project requires NVIDIA GPU. To check whether your GPU is available and is working:

```bash
nvidia-smi
```
If the GPU is working correctly, this command will display detailed information, including driver version, GPU usage, memory usage, and temperature.

Make sure the command `uv` are on your path. Then please clone the repo, set up the virtual environment and run tests:

```bash
# clone the repo
git clone git@github.com:microsoft/SNRAware.git

# set up env
direnv allow
cd ./SNRAware
uv sync

# pull down test data
git lfs pull

# run the test
uv run pytest -m gpu ./test
```

## Training data

Dataset for MR denoising training is not opened at this moment. More information will be provided once training data is released.

## Model
Three models are released at https://huggingface.co/microsoft/SNRAware

- SNRAware-small: 27.7million parameters
- SNRAware-medium: 55.1million parameters
- SNRAware-large: 109million parameters

To test the model, 
```bash
# download the model from the huggingface
# small model
wget https://huggingface.co/microsoft/SNRAware/resolve/main/small/snraware_small_model.pts
wget https://huggingface.co/microsoft/SNRAware/resolve/main/small/snraware_small_model.yaml

# a test data is provided at ./test/data/inference
# input data are [H, W, Frame] 3D complex tensor, input_real.npy and input_imag.npy store the 
# real and imaginary part
# gmap.npy is the g-factor map for all frames or for every frame, [H, W, 1 or Frame]

# let's use the small model to run a inference
export model_file=snraware_small_model.pts
export config_file=snraware_small_model.yaml

# run the inference
uv run python3 ./src/snraware/projects/mri/denoising/run_inference.py --input_dir ./test/data/phantom --output_dir /tmp/phantom_res_inference --saved_model_path $model_file --saved_config_path $config_file --batch_size 1 --input_fname input --gmap_fname gmap
```

After the run, the result is saved in the `/tmp/phantom_res_inference` as numpy files. 

![alt text](./docs/images/image.png)

raw, model output, difference

## Direct intended uses
SNRAware is shared for research and technical development purposes only, to denoisegit  MR images.

## License and Usage Notices
The data, code, and model checkpoints described in this repository is provided for research and technical development use
only. The data, code, and model checkpoints are not intended for use in clinical use.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Documentation

Please find documentation in the [docs/overview](./docs/overview.md).
