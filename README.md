## Important note for setup

The original repo from which this is forked can be found at [https://github.com/asash/bert4rec_repro]. The following instructions are complementary to the ones in that repository, and adapt the setup for both Windows and Linux systems.

## Installation

The instructions has been tested on a Windows 10 machine with an NVIDIA GeForce GTX 1060 6GB.

Please follow these step-by-step instructions to reproduce our results.


### 1. Install Python 3.9.12 in your system.

Install this version of Python on your system and find it's exact path if it's not your default Python.


### 2. Create the project working directory
```
mkdir aprec_repro
cd aprec_repro
```


### 3. Create a python enviroment with the selected python version.

```
python -m venv ".venv"              
```

This will create a hidden folder `.venv` and a virtual enviroment inside of it. If Python3.9.12 isn't your default python version, run the following command instead:

```
<path to you python version> -m venv ".venv"              
```

### 4. Activate your enviroment and make sure the correct python version is active.

Activate the enviroment (on Windows) by using either

```
.venv/Script/activate.bat
```

or

```
.venv/Script/Activate.ps1
```

Make sure the correct version of python is being used by running `pythohn --version` and seeing `3.9.12`.

### 5. Install requirements

First, upgrade pip:

```
pip install --upgrade pip
```

```
pip install -r requirements.txt
```

where `requirements.txt` is [this requirements file](https://github.com/Ocete/TFM/blob/main/aprec_repro/requirements.txt).

### 6. Clone necessary github repos into workdir:

This is step 6 of [https://github.com/asash/bert4rec_repro], but cloning this updated repo instead of their version in step 7.1. To obtain an even cleaner version, clone [this exact commit](https://github.com/Ocete/bert4rec_repro/blob/6fde3e82b0922ef83952a7df6925cdbcb7b6f64a/utils/os_utils.py#L9), right after the setup was compleated and before any other changes were made.


### 7. Download Yelp Dataset if needed
 
Check step 8 in [https://github.com/asash/bert4rec_repro].

### 8. If on Linux, manually change OS setting.

If you are working on Linux instead of Ubuntu, manually change the OS setting in `aprec/utils/os_utils.py` ([here](https://github.com/Ocete/bert4rec_repro/blob/20a1c9d8d98e60b59fe383ba318eae8a4b8f57b7/utils/os_utils.py#L8)) to `False`.

### 9. Test the code

Your environment is now ready to run the experiments. To ensure that everything works correctly, run the tests:

```
cd aprec/tests
pytest --verbose --forked . 
```

If the preivous command doesn't work for running tests (due to Windows incompatibilites with subprocesses), run the following instead:

```
pytest -n auto --verbose . 
```
