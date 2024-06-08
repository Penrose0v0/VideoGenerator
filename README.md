# Video Generator

(This is still an unfinished model)

The main purpose of this repo is to complete my compulsory course project. 


## Introduction

The goal of this model is to generate video frame by frame, based on action prompts. 

By prompting the model with an action and previous frames, the model will generate the future frames of the video. 

At current stage, the architecture of the model will refer to *[GAIA-1](https://arxiv.org/abs/2309.17080)*, a generative world model for autonomous driving. 

At the same time, *[Ginie](https://arxiv.org/abs/2402.15391)* also demonstrated a good performance that I expected, so I'm considering to combine both model together and do experiment on it. 

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