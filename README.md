<h1 align="center">
  <b>Accurate Brain Segmentation in Malformations of Cortical Development</b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.8--3.11-ff69b4.svg" /></a>
      <a href= "https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-1.8--2.4-2BAF2B.svg" /></a>
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

### Using uv (recommended)
```bash
uv add deepmask
```

### Using pip
```bash
pip install deepmask
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
conda create -n deepMask python=3.11
conda activate deepMask
pip install -e .
```

### GPU Installation (CUDA)
For faster inference with CUDA support:

```bash
# Using make (recommended - uses configured sources)
make install-cuda

# Using uv directly (uses configured sources)
uv sync --extra cuda

# Using pip (manual index specification)
uv pip install -e ".[cuda]" --extra-index-url https://download.pytorch.org/whl/cu124/
```

**Requirements:**
- NVIDIA GPU with CUDA support
- Compatible NVIDIA drivers

**Note:** The package sources are configured in `pyproject.toml` to automatically use the correct PyTorch wheels. CPU PyTorch is installed by default, with CUDA available as an optional extra.

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
uv sync --dev --extra cpu
uv run pre-commit install

# Run tests across all supported Python versions
make test-all

# Run integration tests (validates brain mask outputs against reference)
make test-integration

# Format code
make format

# Build package
make build
```

### Modern Development Features

- **UV Package Manager**: Fast, reliable dependency management with lock files
- **Matrix Testing**: Automated testing across Python 3.8-3.11
- **Optimized CI/CD**: GitHub Actions with intelligent caching and artifact sharing
- **Pure Python Wheels**: Universal wheel building for cross-platform compatibility
- **Pre-commit Hooks**: Automated code formatting and linting

### Testing

The project includes comprehensive testing across Python 3.8-3.11:

- **Unit tests**: Fast tests that validate individual components  
- **Integration tests**: End-to-end tests with matrix testing across Python versions
- **CI/CD**: Automated testing with GitHub Actions and optimized caching

```bash
# Run only unit tests (default)
make test

# Run integration tests (requires internet connection for test data)
make test-integration

# Run all tests
make test-all

# Run integration tests manually with script
./run_integration_tests.sh

# Test specific Python version
make test-py311
make test-py310
make test-py39
make test-py38
```

**Integration Test Details:**
- Matrix testing across Python 3.8, 3.9, 3.10, and 3.11
- Downloads test data from OpenNeuro dataset (sub-00055)
- Downloads reference brain mask from deepFCD validation data
- Runs deepMask inference on test subject
- Validates brain mask overlap metrics (Dice > 0.85, Jaccard > 0.75)
- Ensures anatomical consistency and proper segmentation quality
- Artifact sharing between jobs for efficient CI execution

## Requirements
- Python 3.8, 3.9, 3.10, or 3.11
- PyTorch >= 1.8, <2.5.0 (CPU by default, CUDA optional)
- ANTsPy >= 0.4.0
- ANTsPyNet >= 0.2.0 (optional, for additional features)

## License
<a href= "https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" /></a>
```console
Copyright 2025 Neuroimaging of Epilepsy Laboratory, McGill University
```
