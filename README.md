
# OpenSub Bitext Retrieval Fork

This fork was developed for Ontario Tech University's CSCI 6720 course project, focusing on bug fixes and improvements related to the OpenSub bitext Retrieval Dataset.

**Roleplaying Seminar 1**  
Implementer: Zikun Fu<br>

A presentation on this project can be found here: [Link](https://docs.google.com/presentation/d/15_714D-FxXXxpVXP4rQg-a9k0smK_srcMI1aEZalz9Q/edit?usp=sharing)

## 🔧 Environment Setup

This project uses `conda` to manage the environment. To set up the required environment, run the following command:

```bash
conda env create -f environment.yml
conda activate miners
```

**Microsoft Visual C++ 14.0 or greater** is required to run the project. You can download it from [Microsoft's website](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

## 📝 Experiment Logs

Experiment logs can be found in the Jupyter notebook provided [here](./logs.ipynb).

## 🚀 Running Experiments

### Bitext Retrieval

The main focus of this project is the bitext retrieval task using the OpenSub dataset.<br> 
You can run experiments by following the instructions in the provided notebooks and scripts.
```
❱❱❱ python bitext.py --src_lang af --dataset opensub --seed 42 --cuda --model_checkpoint sentence-transformers/LaBSE
```

## 📜 Credits

- **OpenSub bitext mining dataset**: [Loïc Magne's OpenSubtitles Dataset](https://huggingface.co/datasets/loicmagne/open-subtitles-bitext-mining)
- Original benchmark code based on the MINERS framework.<pre>
@article{winata2024miners,
  title={MINERS: Multilingual Language Models as Semantic Retrievers},
  author={Winata, Genta Indra and Zhang, Ruochen and Adelani, David Ifeoluwa},
  journal={arXiv preprint arXiv:2406.07424},
  year={2024}
}
</pre>