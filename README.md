# HarryGPT: Custom trained LLM models
This repo features scripts for training small LLM models in `training/` and a flask-based webserver for interacting with these models in `server/`.

For running the code in this repo, a python environment with pytorch is required.

1. Create a venv (e.g. in vscode: 'Python: Select Interpreter')
2. Install pytorch
  - cuda: `pip install numpy torch --index-url https://download.pytorch.org/whl/cu121`
  - cpu:  `pip install numpy torch --index-url https://download.pytorch.org/whl/cpu`
3. Install training dependencies: `pip install matplotlib optuna`
4. Install server   dependencies: `pip install flask`

src: https://pytorch.org/get-started/locally

Now, run any training script from within `training/`. Results will be stored in `training/results`.

Run the server from within `server/` with `python server.py` and visit `http://localhost:5000` to iteract with it.
The server will provide models from the `server/models` folder, so make sure to copy models from `training/results` that you want to keep.

The server can also be run as a Docker container. For this, run:

1. `sudo docker build -t harry-gpt .`
2. `sudo docker run -p 8080:5000 harry-gpt`

or integrate this project in a docker-compose file.
