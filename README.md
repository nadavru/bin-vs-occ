# Binary vs OCC

The code implementation for "One-Class Classifier with Two-Sample Testing":

Project by Nadav Rubinstein and Arnon J. Kidron.

Supervisor: Yaniv Romano

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### For synthetic data:

For all the datasets and all the outliers' ratios:
```bash
python main_exp_synth.py
```

For specific dataset and outliers' ratio (ind from 0,...):
```bash
python main_exp_synth.py <ind>
```

For specific dataset and outliers' ratio with specific device:
```bash
python main_exp_synth.py <ind> <device>
```

### For real data:

For all the datasets:
```bash
python main_exp_real.py
```

For specific dataset (ind from 0,...):
```bash
python main_exp_real.py <ind>
```

For specific dataset with specific device:
```bash
python main_exp_real.py <ind> <device>
```

### For real data with bootstrapping for smaller outlier's sample:

For all the datasets and all the outliers' ratios:
```bash
python main_exp_real2.py
```

For specific dataset and outliers' ratio (ind from 0,...):
```bash
python main_exp_real2.py <ind>
```

For specific dataset and outliers' ratio with specific device:
```bash
python main_exp_real2.py <ind> <device>
```

## References

Datasets taken from official ODDS site:
http://odds.cs.stonybrook.edu/
