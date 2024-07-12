# Video Generator

(This is still an unfinished model)

The main purpose of this repo is to complete my compulsory course project. 


## Introduction

The goal of this model is to generate video frame by frame, based on action prompts. 

By prompting the model with an action and previous frames, the model will generate the future frames of the video. 

At current stage, the architecture of the model will refer to *[GAIA-1](https://arxiv.org/abs/2309.17080)*, a generative world model for autonomous driving. 

By the way, *[Genie](https://arxiv.org/abs/2402.15391)* also demonstrated a good performance that I expected, so I'm considering to combine both models together and do experiment on it. 


## Installation
### Basic
Simply, run: 
```
pip install -r requirements.txt
```
In addition, the environment is built with Python 3.9 and CUDA 11.8

If you failed to install dependencies in this way, please run the following command: 
```
pip install matplotlib
pip install opencv-python opencv-contrib-python
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### MineRL
If you need to generate the Minecraft random video data, you have to install MineRL. 

First of all, install jdk8: 
```
add-apt-repository ppa:openjdk-r/ppa
apt-get update
apt-get install openjdk-8-jdk

# Verify installation
java -version # this should output "1.8.X_XXX"
# If you are still seeing a wrong Java version, you may use the following line to update it
# sudo update-alternatives --config java
```

Then, install MineRL: 
```
pip install git+https://github.com/minerllabs/minerl
```


## Train
Basically, run: 
```
python train.py
```
Please read the code of ```train.py``` for more details.


## Citations

```bibtex
@article{hu2023gaia,
  title={Gaia-1: A generative world model for autonomous driving},
  author={Hu, Anthony and Russell, Lloyd and Yeo, Hudson and Murez, Zak and Fedoseev, George and Kendall, Alex and Shotton, Jamie and Corrado, Gianluca},
  journal={arXiv preprint arXiv:2309.17080},
  year={2023}
}
```

```bibtex
@article{bruce2024genie,
  title={Genie: Generative Interactive Environments},
  author={Bruce, Jake and Dennis, Michael and Edwards, Ashley and Parker-Holder, Jack and Shi, Yuge and Hughes, Edward and Lai, Matthew and Mavalankar, Aditi and Steigerwald, Richie and Apps, Chris and others},
  journal={arXiv preprint arXiv:2402.15391},
  year={2024}
}
```