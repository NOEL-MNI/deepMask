<h1 align="center">
  <b>Accurate Brain Segmentation in Malformations of Cortical Development</b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.8+-ff69b4.svg" /></a>
      <a href= "https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-1.8%20LTS-2BAF2B.svg" /></a>
      <a href= "https://github.com/NOEL-MNI/deepMask/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" /></a>
      <a href="https://doi.org/10.5281/zenodo.4521706">
        <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4521706.svg" alt="DOI"></a>


</p>

PyTorch Implementation using V-net variant of Fully Convolutional Neural Networks

Authors: [Ravnoor Gill](https://github.com/ravnoor), [Benoit Caldairou](https://github.com/bcaldairou), [Neda Bernasconi](https://noel.bic.mni.mcgill.ca/~noel/people/neda-bernasconi/) and [Andrea Bernasconi](https://noel.bic.mni.mcgill.ca/~noel/people/andrea-bernasconi/)

------------------------

![](assets/diagram.png)

Implementation based on:<br>
Milletari, F., Navab, N., & Ahmadi, S. A. (2016, October). [V-net: Fully convolutional neural networks for volumetric medical image segmentation](https://arxiv.org/abs/1606.04797). In 2016 Fourth International Conference on 3D vision (3DV) (pp. 565-571). IEEE.


### Please cite:
```TeX
@misc{Gill2021,
  author = {Gill RS, et al},
  title = {Accurate and Reliable Brain Extraction in Cortical Malformations},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/NOEL-MNI/deepMask}},
  doi = {10.5281/zenodo.4521716}
}
```

## Installation

### Using pip (recommended)
```bash
pip install deepmask
```

### Using uv (fastest)
```bash
uv add deepmask
```

### Development Installation
```bash
git clone https://github.com/NOEL-MNI/deepMask.git
cd deepMask
uv sync --dev
```

### Traditional Installation
```bash
git clone https://github.com/NOEL-MNI/deepMask.git
cd deepMask
conda create -n deepMask python=3.8
conda activate deepMask
pip install -e .
```

### GPU Installation (CUDA 11.1)
For faster inference with CUDA support:

```bash
# Using make (recommended - uses configured sources)
make install-cuda

# Using uv directly (uses configured sources)
uv pip install -e ".[cuda]"

# Using pip (manual index specification)
pip install -e ".[cuda]" --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111/
```

**Requirements:**
- NVIDIA GPU with CUDA 11.1 support
- NVIDIA drivers compatible with CUDA 11.1

**Note:** The package sources are configured in `pyproject.toml` to automatically use the correct PyTorch wheels based on the installation option.

## Usage

### Command Line Interface
```bash
deepmask-inference patient_id t1.nii.gz flair.nii.gz /output/directory
```

### Python API
```python
from deepmask import vnet
from deepmask.utils.deepmask import noelImageProcessor

# Load model
model = vnet.build_model(args)

# Process images
processor = noelImageProcessor(
    id="patient_001",
    t1="path/to/t1.nii.gz",
    t2="path/to/flair.nii.gz",
    output_dir="/output/path",
    model=model
)
processor.pipeline()
```

### Docker Usage
```bash
docker run -it -v /tmp:/tmp docker.pkg.github.com/noel-mni/deepmask/deepmask:latest \
    deepmask-inference patient_id /tmp/T1.nii.gz /tmp/FLAIR.nii.gz /tmp
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

### Quick Development Setup
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/NOEL-MNI/deepMask.git
cd deepMask
uv sync --dev
uv run pre-commit install

# Run tests
make test

# Run integration tests (validates brain mask outputs against reference)
make test-integration

# Run all tests including integration tests
make test-all

# Format code
make format

# Build package
make build
```

### Testing

The project includes both unit tests and integration tests:

- **Unit tests**: Fast tests that validate individual components
- **Integration tests**: End-to-end tests that validate brain mask outputs against reference data from deepFCD

```bash
# Run only unit tests (default)
make test

# Run integration tests (requires internet connection for test data)
make test-integration

# Run all tests
make test-all

# Run integration tests manually with script
./run_integration_tests.sh
```

**Integration Test Details:**
- Downloads test data from OpenNeuro dataset (sub-00055)
- Downloads reference brain mask from deepFCD validation data
- Runs deepMask inference on test subject
- Validates brain mask overlap metrics (Dice > 0.85, Jaccard > 0.75)
- Ensures anatomical consistency and proper segmentation quality

## Requirements
- Python >= 3.8
- PyTorch >= 1.8.0
- ANTsPy >= 0.4.0
- ANTsPyNet >= 0.2.0

## License
<a href= "https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" /></a>
```console
Copyright 2021 Neuroimaging of Epilepsy Laboratory, McGill University
```
