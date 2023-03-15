# README #
Using deep learning, we recognize different kind of textiles and classify captured images into valid and invalid groups.

A deep learning approach and pytorch has been chosen to build the model.
scikit-learn has been used to calculate some classification metrics like accuracy, precision, recall and f1_score.
It also have been used to plot ROC and precision-recall curves, and also for plotting confusion matrices.
You can find all those metrics in the mlruns folder. 

Three famous deep models named efficientnet_v2_s, densenet121, resnext50_32x4d were adopted, changed and fine-tuned base on our task.

MLFlow has been chosen to log parameters, metrics, checkpoints, figures during training and testing.
You can find the logs for each model in the mlruns directory.
To see the logs, you need to run "mlflow ui" after you installed the package.

Some functions and classes have been tested using Unittest. You can find them in the tests directory.

In the project directory, you can find Makefile to create the virtual environment and required packages automatically.

The data have been included in the project directory, which is not a good thing. 
It is better to use Data Version Control to keep data and artifacts somewhere else without having to put them on github, but I wanted to keep everything in one place, so no dvc was used.

There is a directory named hub which holds pre-trained models and fine-tuned models.

You can find the html files of the documentation in the directory name site. If you open the index.html file you can walk through the documentation
There is a directory named docs, you can find .md files corresponding to each .py files which are needed to generate the html files.

Other kinds of logs like error for monitoring the training will be recorded in a directory named logs, and if something goes wrong you can find it there.

In order to be able of using this package, you need to follow these steps.

# Install prerequisites: #
## install python3: ##
* It doesn't matter which version of python you chose as long as it is newer than 3.6 since we will install python 3.9.12 locally using pyenv via makefile

## install pyenv to set up the correct version of Python: ##
* For windows operating system, install pyenv-win by reading this link: https://github.com/pyenv-win/pyenv-win
* Or Open PowerShell and run these commands:
* pip install pyenv-win --target $HOME\\.pyenv
* [System.Environment]::SetEnvironmentVariable('PYENV',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")
* [System.Environment]::SetEnvironmentVariable('PYENV_ROOT',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")
* [System.Environment]::SetEnvironmentVariable('PYENV_HOME',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")
* [System.Environment]::SetEnvironmentVariable('path', $env:USERPROFILE + "\.pyenv\pyenv-win\bin;" + $env:USERPROFILE + "\.pyenv\pyenv-win\shims;" + [System.Environment]::GetEnvironmentVariable('path', "User"),"User")



* for linux you need to run following commands on your terminal (https://github.com/pyenv/pyenv):
* sudo apt-get update
* sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
* git clone https://github.com/pyenv/pyenv.git ~/.pyenv
* echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
* echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
* echo 'eval "$(pyenv init -)"' >> ~/.bashrc
* Then, if you have ~/.profile, ~/.bash_profile or ~/.bash_login, call the there echo command up there, but replace ~/.bashrc by ~/.profile, ~/.bash_profile or ~/.bash_login
* If you have none of these, stream the three echo commands to ~/.profile.
* Bash warning: There are some systems where the BASH_ENV variable is configured to point to .bashrc. On such systems, you should almost certainly put the eval "$(pyenv init -)" line into .bash_profile, and not into .bashrc. Otherwise, you may observe strange behaviour, such as pyenv getting into an infinite loop
* exec "$SHELL"


## install make to automate commands: ##
* For windows operating system: install make using chocolatey: https://chocolatey.org/install or run following command on the powersell as admin: 
* Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
* Then run following command on windows powersell as admin: 
* choco install make


* for linux just call following commands:
* sudo apt-get update
* sudo apt-get install make


# Use the Package in following way: ##
change your path on the terminal to the project directroy. 

run this command to see makefile's help: 
* make help

run this command to create virtual environment and install all required packages: 
* make venv

activate the virual environment by running this command (required step):
* .\venv\Scripts\activate on Windows
* source ./venv/bin/activate on Linux

if you want to run the test cases, use this command: 
* make test

run this command to train the model. you can also see its manual before running it.
* python .\textile_classification\train.py --help
* python .\textile_classification\train.py .\textile_classification\train_input_info.cfg

run this command to evaluate the model when you know the ground truth. you can also see its manual before running it:
* python .\textile_classification\test.py --help 
* python .\textile_classification\test.py .\textile_classification\test_input_info.cfg

run this command to do prediction when you don't know the ground truth. you can also see its manual before running it:
* python .\textile_classification\predict.py --help
* python .\textile_classification\predict.py .\textile_classification\predict_input_info.cfg

## Note: 
You can find the parameters, metrics and artifacts in mlruns direcory. Run this command on your terminal to see them:
* mlflow ui

you can find already created html files of the documentation in site directory, 
but if you change the code and you want to create new documentation you can do it temporaray and permanetly by running following commands:
* python -m mkdocs serve
* python -m mkdocs build


