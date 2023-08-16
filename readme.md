# Poli-baselines: a series of baselines for discrete sequence optimization

`poli-baselines` is a collection of **black box optimization algorithms**, aimed mostly at optimizing discrete sequences. These optimization algorithms are meant to optimize objective functions defined using [`poli`](https://github.com/MachineLearningLifeScience/poli), a tool for isolating complex, difficult-to-query functions.

## Installation

Create a fresh conda environment by running

```bash
conda create -n poli-baselines python=3.9
conda activate poli-baselines
```

After which, you can download this repository and install it locally

```bash
git clone git@github.com:MachineLearningLifeScience/poli-baselines.git
cd ./poli-baselines
```

Now install the requirements, as well as the library:

```bash
pip install -r requirements.txt
pip install -e .
```

After this, you could test you installation by running (inside your `poli-baselines` environment):

```bash
python -c "import poli_baselines ; print('Everything went well!')"
```

## Where to find documentation?

You can find documentation about both `poli` and `poli-baselines` on [TODO:Add the website when we publish it].

This documentation is built from this repository: https://github.com/MachineLearningLifeScience/protein-optimization-docs

**FOR NOW, AND UNTIL WE PUBLISH THE WEBSITE AND THIS REPO:**

Since we haven't published this documentation,

1. clone that repo:

```bash
git clone git@github.com:MachineLearningLifeScience/protein-optimization-docs.git
```

2. Create a `poli-docs` environment (if you want).

```bash
conda create -n poli-docs python=3.9
conda activate poli-docs
```

3. Install the requirements:

```bash
pip install jupyter-book biopython pandas
```

4. Build the docs locally:

```bash
jupyter-book build --all docs/protein-optimization
```

5. Open the relevant address in your browser (check the output of the last step).

[TODO: remove this when we publish the website]

