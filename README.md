## Important note for setup

The original repo from which this is forked can be found at https://github.com/asash/bert4rec_repro. The following instructions are complementary to the ones in that repository, and adapt the setup for both Windows and Linux systems.

## Installation

The instructions has been tested on a Windows 10 machine with an NVIDIA GeForce GTX 1060 6GB.

To complete the installation, follow each step of these instructions. If there is a step missing (for instance, like steps 1 and 2), follow the missing steps from the [original repo](https://github.com/asash/bert4rec_repro) instead.

### 3. Create an anaconda environment with necessary package versions:

Install each package version following the official [tensorflow dependency matrix](https://www.tensorflow.org/install/source#tested_build_configurations) instead of the ones from the original repo.

```
conda create -y -c pytorch -c conda-forge --name aprec_repro python=3.9.12 cudnn=8.1 cudatoolkit=11.2
conda install gh=2.1.0 
```

Note: Installing `expect` like in the original repo is only necessary for Linux machines.

Activate the conda enviroment now in order to install packages with pip inside the enviroment. You might need to run `conda init` or `conda init powershell` first:

```
conda activate aprec_repro
```

Instead of installing tensorflow or pytorch via conda (like `conda install tensorflow-gpu=2.6.2`) use pip, once the conda enviroment has been activated.

```
pip install tensorflow==2.6.0
```

Torch may be also installed using pip, but it's already on the requirements list, so it will be installed in step 6.

### 4. Add working working directory to the PYTHONPATH of the anaconda environment:

Instead of running:

```
conda env config vars set -n aprec_repro PYTHONPATH=`pwd`
```

You will need to add the path to the new `python.exe` executable (inside the conda directory) manually, and use `'` instead of ` ` `. In order to easily find where the python executable is, you can run (once the conda enviroment is active):

```
python
> import sys
> sys.path
```

In my case the exact path was `D:\anaconda3\envs\aprec_repro\python.exe` so I ran:

```
conda env config vars set -n aprec_repro PYTHONPATH='D:\anaconda3\envs\aprec_repro\python.exe'
```

### 5. Install python packages in the environment

Our conda enviroment should be already active :)

### 6. Install python packages in the environment

Use this [requirements file](https://github.com/Ocete/TFM/blob/main/aprec_repro/requirements.txt) and run:

```
pip install -r requirements.txt
```

This will install not only the necessary packages, but the exact old versions compatible with each other.

### 7. Clone necessary github repos into workdir:

Follow the other instructions but clone this updated repo instead of their version in step 7.1. To obtain an even cleaner version, clone [this exact commit](https://github.com/asash/bert4rec_repro/commit/3f2959da7f4d230cd7e4a734ffbd42e912aa0618), right after the setup was compleated and before any other changes were made.

#### 7.3 Code adaptation 

If you are working on Linux instead of Ubuntu, manually change the OS setting in `aprec/utils/os_utils.py` ([here](https://github.com/Ocete/bert4rec_repro/blob/20a1c9d8d98e60b59fe383ba318eae8a4b8f57b7/utils/os_utils.py#L8)) to `False`.

In that same file, change the variable `python_exec_path` to your `python.exe` (the same path used in step 4 but with double backslash).

### 9. Test the code

Your environment is now ready to run the experiments. To ensure that everything works correctly, run the tests:

```
cd aprec/tests
pytest --verbose --forked . 
```

If the preivous command doesn't work for running tests (due to Windows incompatibilites with subprocesses), run the following instead:

```
pytest --verbose -n auto . 
```

### 10. Using the code to launch experiments

In order to launch experiments we will use the script `evaluation/run_n_experiments.sh`, which needs some further tweaking before working:

- It must be updated with your location of the conda enviroment `python.exe` file (when using the command `python` in a script on Windows, the conda python wouldn't be used for some reason). Just manually add the path around line 40.
- Make sure to save this single file with LF instead of CRLF. This can easily be done in VSCode.
- Install the necessary tools to be able to run the `bash` command from the powershell.

To launch experiments, go to the evaluation folder and use:

```
bash run_n_experiments.sh configs/<my config>
```