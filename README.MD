
## Setup
To install the necessary dependencies, run the following commands:
```
conda create --name dcd python=3.8
conda activate dcd
pip install -r requirements.txt
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
cd ..
pip install pyglet==1.5.11
```
place minigrid.py inside gym_minigrid folder of conda env 

