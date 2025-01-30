# Use the official Miniconda image as a parent image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /usr/src/app

# Create and configure the Conda environment
RUN conda update -n base -c defaults conda && \
    conda create --name mario python=3.8 -y && \
    echo "source activate mario" > ~/.bashrc && \
    conda install -c conda-forge -c pytorch cudatoolkit=11.6 torchvision torchaudio -y && \
    conda update -n base -c defaults conda -y

ENV PATH /opt/conda/envs/mario/bin:$PATH

# Install necessary Debian packages and Python packages
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --upgrade pip setuptools wheel && \
    pip install nes_py gym==0.17.2 gym_super_mario_bros==7.3.0 opencv-python matplotlib && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Jupyter
RUN pip install jupyter

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run Jupyter Notebook when the container launches
CMD ["jupyter", "notebook", "690_final_project.ipynb", "--ip='*'", "--port=8888", "--allow-root", "--NotebookApp.open_browser=True"]
