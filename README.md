# Q-CHIPP

Q-CHIPP (Quantum Convolutional HLA Immunogenic Peptide Prediction) is a combinatorial framework integrating MHC binding and T-cell recognition to more accurately identify immunogenic peptides, improving the prognostic impact of predicted neoantigen load.

## Citation

Please cite the following article if you use Q-CHIPP:

**Quantum Convolutional HLA Immunogenic Peptide Prediction (Q-CHIPP): Next-Generation Neoantigen Prediction with Quantum Neural Networks**  
Ryan Peters, Kahn Rhrissorrakrai, Prerana Bangalore Parthasarathy, Vadim Ratner, Tanvi P. Gujarati, Meltem Tolunay, Jie Shi, Jeffrey K. Weber, Timothy A. Chan, Laxmi Parida, Sara Capponi, Filippo Utro, Tyler J. Alban.  
doi: https://doi.org/10.1101/2025.07.29.667313

## Requirements

- Python 3.12
- Qiskit 2.x
- See `requirements.txt` for complete list of dependencies

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Q-CHIPP/
```

### 2. Create Virtual Environment
It is recommended to create a Python virtual environment:

```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```


## Running Experiments

### Basic Usage

To run experiments with default configuration:

```bash
python qcnn_all.py --config-name=defaults_qcnn.yaml
```

### Custom Configuration

1. Create a new YAML configuration file in the `configs/` directory
2. Specify all relevant parameters (see Parameter Descriptions below)
3. Run with your custom config:

```bash
python qcnn_all.py --config-name=your_config.yaml
```

## Configuration Parameters

Q-CHIPP uses [Hydra](https://hydra.cc/) for configuration management. Key parameters include:

### Data Parameters
- `train_data_file`: Path to training data CSV
- `test_data_file`: Path to test data CSV
- `encoding_method`: Data encoding method (e.g., 'binary', 'ordinal')

### Model Parameters
- `feature_map`: Type of feature map ('ZFeatureMap', 'ZZFeatureMap', 'qrac_21', 'qrac_31', 'basis')
- `ansatz`: Ansatz type for variational circuit
- `primitive`: Qiskit primitive to use ('estimator' or 'sampler')
- `backend`: Backend for execution ('statevector', 'aer_simulator', or IBM backend)

### Training Parameters
- `optimizer`: Optimization algorithm ('COBYLA', 'SPSA', 'L_BFGS_B')
- `max_iter`: Maximum iterations for optimizer
- `shots`: Number of measurement shots (for hardware/simulator)
- `seed`: Random seed for reproducibility

### Output Parameters
- `dir_output`: Output directory for results
- `file_output`: Output filename for results

See `configs/defaults_qcnn.yaml` for a complete example with all available parameters.

## Notebooks

The `notebooks/` directory contains Jupyter notebooks for:
- Data preparation and preprocessing
- Reading and analyzing experiment outputs
- Creating summary dataframes of results
- Programmatically generating YAML configurations


## Troubleshooting

### Common Issues

**Import Errors**: Ensure you're using Python 3.12 and have activated the virtual environment.

**Memory Issues**: For large datasets, consider reducing batch size or using a machine with more RAM.

**Backend Connection Issues**: For IBM Quantum backends, ensure your credentials are properly configured.

## Output

Results are saved to the `experiments/` directory (or custom output directory specified in config) including:
- Model weights (`.npy` files)
- Objective function values
- Classification results (`.pkl` files)

## License

See LICENSE file for details.

## Support

For issues, questions, or contributions, please refer to the citation paper or contact the authors.
