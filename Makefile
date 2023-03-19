VENV_DIR = venv
PROXY_FLAG = 
TEST_DIR = tests
PYTORCH_REPOSITORY = 

# Help
.PHONY: help
help:
	@echo "Variables:"
	@echo "	VENV_DIR:" 
	@echo "		virtual environment's folder name"
	@echo "		This is an example to change VENV_DIR: make venv VENV_DIR=my_venv    "
	@echo "	PROXY_FLAG:" 
	@echo "		If you wanna use the application on a computer having proxy,"
	@echo "		you have to determine PROXY_FLAG when you call make command." 
	@echo "		For example: make venv PROXY_FLAG='--proxy http://<usr_name>:<password>@<proxyserver_name>:<port#>'" 
	@echo "		Otherwise, don't change it or determine it."
	@echo "	PYTORCH_REPOSITORY:"
	@echo "		It is used to install pytorch compatible with your cuda version."
	@echo "		Pass correct url regarding your cuda version."
	@echo "		For example: make venv PYTORCH_REPOSITORY='--extra-index-url https://download.pytorch.org/whl/cu116'"
	@echo "		Before that you need to change cuda version in the requirements.txt file."
	@echo "		for example, put the cuda version there like this: torch==1.13.1+cu116, torchvision==0.14.1+cu116"
	@echo "		If you don't have cuda installed, you don't need to change it or determine it."
	
	@echo "Commands:"
	@echo "	venv: creates a virtual environment in the folder named venv."
	@echo "	test: execute all tests cases located in the folder named tests."
	@echo "	clean: cleans all unnecessary files."

# Environment
.PHONY: venv
ifeq (${OS},Windows_NT)
venv:
	pyenv install 3.9.12
	pyenv local 3.9.12
	python -m venv ${VENV_DIR}
	${VENV_DIR}\Scripts\python -m pip install ${PROXY_FLAG} pip setuptools wheel
	${VENV_DIR}\Scripts\python -m pip install ${PROXY_FLAG} --upgrade pip setuptools wheel
	${VENV_DIR}\Scripts\python -m pip install ${PROXY_FLAG} -e . ${PYTORCH_REPOSITORY}
else
venv:
	pyenv install 3.9.12
	pyenv local 3.9.12
	python -m venv ${VENV_DIR}
	${VENV_DIR}/bin/python -m pip install ${PROXY_FLAG} pip setuptools wheel
	${VENV_DIR}/bin/python -m pip install ${PROXY_FLAG} --upgrade pip setuptools wheel
	${VENV_DIR}/bin/python -m pip install ${PROXY_FLAG} -e . ${PYTORCH_REPOSITORY}
	
endif

# Test
.PHONY: test
ifeq (${OS},Windows_NT)
test:
	${VENV_DIR}\Scripts\python -m unittest discover ${TEST_DIR}
else
test:
	${VENV_DIR}/bin/python -m unittest discover ${TEST_DIR}
	
endif

# Clean
.PHONY: clean
ifeq (${OS},Windows_NT)
clean:
	@echo "Note: Cleaning process has not been implemented for Windows Operating Systems."
else
clean:
	rm -rf __pycache__
	rm -rf textile_classification/__pycache__
	rm -rf tests/__pycache__
	rm -rf textile_classification/data_setters/__pycache__
	rm -rf textile_classification/utils/__pycache__
	rm -rf textile_classification/model/__pycache__
	rm -rf hub/pretrained_models/__pycache__
	rm -rf ${VENV_DIR}
	rm -rf *.egg-info/
	rm -rf .python-version
endif
