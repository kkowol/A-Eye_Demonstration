# A-Eye: Demonstration

This repo is based on the CARLA 0.9.10 code and was adapted for our research (https://carla.readthedocs.io/en/0.9.10/). 

In addition, this repo contains the code from: https://github.com/kkowol/A-Eye, where we use the A-Eye method to quickly detect corner cases, as well as adaptations from the following article: https://link.springer.com/chapter/10.1007/978-3-031-41962-1_8.

The content of these two works provide the content for this demonstrator with the goal: \
\
**Driving with the eyes of AI**


<!-- # A-Eye: Demonstration
This repo provides real-time driving on the output of a semantic segmentation network. Using it, a method was developed to collect synthetic corner cases, especially those that are difficult for AI algorithms to detect, in a relatively short time with the help of two human drivers. The results were published in our paper "A-Eye: Driving with the Eyes of AI for Corner Case Generation" (https://arxiv.org/abs/2202.10803) -->

## Structure

```bash
├── models          # usable models are placed here
├── supplement      # additional scripts
├── utils           # required scripts 
└── weights         # trained model weights
```

## Installation
First you need a working CARLA 0.9.10 version on your system. For this follow the instructions on the official CARLA repo for version 10: https://carla.readthedocs.io/en/0.9.10/


It is recommended to create a python3 environment and install all required packages. Our code runs with python version 3.6
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
Also you should create an output folder in your working directory:
```bash
mkdir output
```
## Run Code
Now you can start with the first test drive: 
```bash
python main.py
```
The keys 2-5 contain the inference sensor with 4 different weights for the Fast-SCNN. 2 represents the basis for the corner cases and 3,4,5 the trained models with the data sets original, corner cases and pedestrian enriched.

Preset weather conditions can be selected using the F5 to F8 keys (F5=clear, F6=rain, F7=fog, F8=night). 

## Citation
If you find our work useful for your research, please cite our papers:
```
@conference{kowol2022,
author={Kamil Kowol. and Stefan Bracke. and Hanno Gottschalk.},
title={A-Eye: Driving with the Eyes of AI for Corner Case Generation},
booktitle={Proceedings of the 6th International Conference on Computer-Human Interaction Research and Applications - CHIRA,},
year={2022},
pages={41-48},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0011526500003323},
isbn={978-989-758-609-5},
issn={2184-3244},
}

@InProceedings{kowol2023,
author={Kowol, Kamil and Bracke, Stefan and Gottschalk, Hanno},
editor={Holzinger, Andreas and da Silva, Hugo Pl{\'a}cido and Vanderdonckt, Jean and Constantine, Larry},
title={survAIval: Survival Analysis with the Eyes of AI},
booktitle={Computer-Human Interaction Research and Applications},
year={2023},
publisher={Springer Nature Switzerland},
address={Cham},
pages={153--170},
isbn={978-3-031-41962-1},
}
```
