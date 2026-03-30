# Define the name of your conda environment here
ENV_NAME=.jax_conda_env_$(DIR)

# Path to the Conda executable
CONDA=conda

# Default target executed when no arguments are given to make.
default: create install

install_conda : 
	@mkdir -p ~/miniconda3
	@wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
	@bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
	@rm -rf ~/miniconda3/miniconda.sh
	~/miniconda3/bin/conda init bash

# Target to create a new conda environment from the environment.yml file
create:
	@echo "Creating the conda environment..."
	$(CONDA) create --name $(ENV_NAME) python=3.12 -y

# Target to install additional Python packages from a list in a file
install: create
	@echo "Installing additional packages from py_req.txt into the conda environment..."
	$(CONDA) run -n $(ENV_NAME) pip install --upgrade pip

	$(CONDA) run -n $(ENV_NAME) pip install --upgrade jax==0.4.25 jaxlib==0.4.25+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

	$(CONDA) run -n $(ENV_NAME) pip install flax==0.8.0 --no-deps
	$(CONDA) run -n $(ENV_NAME) pip install optax==0.1.8 --no-deps
	$(CONDA) run -n $(ENV_NAME) pip install orbax-checkpoint==0.5.0  --no-deps
	$(CONDA) run -n $(ENV_NAME) pip install numpy==1.26.3 --no-deps
	$(CONDA) run -n $(ENV_NAME) pip install notebook
	$(CONDA) run -n $(ENV_NAME) pip install ipykernel
	$(CONDA) run -n $(ENV_NAME) pip install torch==2.3.0 torchaudio==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118
	$(CONDA) run -n $(ENV_NAME) pip install torchtext==0.17.2
	$(CONDA) run -n $(ENV_NAME) python3 -m ipykernel install --user --name=$(ENV_NAME) --display-name "$(ENV_NAME)"
	

.PHONY: create install
all: install

activate:
	@echo "Activating the conda environment..."
	@echo "Note: You cannot activate a conda environment from a Makefile. Please run 'conda activate $(ENV_NAME)' manually."

# Target to delete the conda environment
clean:
	@echo "Removing the conda environment..."
	$(CONDA) env remove -n $(ENV_NAME)

.PHONY: default create install
