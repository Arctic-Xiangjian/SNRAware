# SNRAware

This repository contains the Pytorch code in our paper [SNRAware: Improved Deep Learning MRI Denoising with Signal-to-noise Ratio Unit Training and G-factor Map Augmentation](https://pubs.rsna.org/doi/full/10.1148/ryai.250227) published at the Radiology: Artificial Intelligence:

```latex
@article{
    doi:10.1148/ryai.250227,
    author = {Xue, Hui and Hooper, Sarah M. and Pierce, Iain and Davies, Rhodri H. and Stairs, John and Naegele, Joseph and Campbell-Washburn, Adrienne E. and Manisty, Charlotte and Moon, James C. and Treibel, Thomas A. and Hansen, Michael S. and Kellman, Peter},
    title = {SNRAware: Improved Deep Learning MRI Denoising with Signal-to-noise Ratio Unit Training and G-factor Map Augmentation},
    journal = {Radiology: Artificial Intelligence},
    volume = {0},
    number = {ja},
    pages = {e250227},
    year = {0},
    doi = {10.1148/ryai.250227},
    note ={PMID: 41123451},
    URL = {https://doi.org/10.1148/ryai.250227}
}
```

- Model type: Imaging AI, non-generative
- License: MIT

## Get started

[just](https://github.com/casey/just) is used in this project. If not, please install this tool:

```bash
# install just
wget -qO - 'https://proget.makedeb.org/debian-feeds/prebuilt-mpr.pub' | gpg --dearmor | sudo tee /usr/share/keyrings/prebuilt-mpr-archive-keyring.gpg 1> /dev/null
echo "deb [arch=all,$(dpkg --print-architecture) signed-by=/usr/share/keyrings/prebuilt-mpr-archive-keyring.gpg] https://proget.makedeb.org prebuilt-mpr $(lsb_release -cs)" | sudo tee /etc/apt/sources.list.d/prebuilt-mpr.list
sudo apt update
sudo apt install just -y
```

Then, please set up the virtual environment and run tests:

```bash
# show the list
just --list

# set up virtual environment
just setup-env

# review documentation
just serve-docs

# run test
just test
```

## Data
Dataset for MR denoising training is not opened at this moment.

## Model
Three models are released at https://huggingface.co/microsoft/SNRAware

- SNRAware-small: 27.7million parameters
- SNRAware-medium: 55.1million parameters
- SNRAware-large: 109million parameters

## Direct intended uses
SNRAware is shared for research and technical development purposes only, to denoise MR images.

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
