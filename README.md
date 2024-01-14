# Adversarial Attack on Authorship Identification

This project studies adversarial attack on authorship identification model in various ways.

Introduction sildes can be accessed at [here](https://www.canva.com/design/DAF3y7AtHxw/rdpKnRB2FiBPmdHh_YVvUg/view?utm_content=DAF3y7AtHxw&utm_campaign=designshare&utm_medium=link&utm_source=editor).

Adversarial Attack is the type of attack which aims to
confuse the classifier by modifying a small part of input data,
makes it barely unable to notice the modification by human
but successfully tricked the classifier. In our work, we testify the tolerance of adversarial attack on selected existing Natural Language Processing (NLP) model. We choose authorship identification classifier in our work.

We implemented five different attacking approaches, with the attack can be separated into two different phases:

- Phase 1: Confusion. In phase one, our goal is to confuse
the classifier so the modified articles should be identified
as any other authors’ creation excepts the original one.
- Phase 2: Assignment. In phase two, our goal is to make
every article being identified as articles from a specific
author.

The baseline (opponent) model can be found at [this github repo](https://github.com/arthtalati/Deep-Learning-based-Authorship-Identification/tree/master).

## Repo Structure

In main branch, our most successful approach in stored in folder *phase 1* and *phase 2* for two different phases respectively. They used the following methods:

- Phase 1: choose candidates by POS tagging with the help of `nltk` library and replace with misspelling.
- Phase 2: use enhanced TF-IDF to filter out candidates and use Genetic Algorithm (GA) to optimize our result.

As for other three different branches, they represent three different attempts with the following naming rule:

$$(\text{method to select candidates to be replaced}) / (\text{method to find alternatives for replacing candidates})$$


## Setup

For attacking models in main branch, we built those models on *Kaggle*. The information about our *Kaggle* enviroment and some brief guides is on 
[https://www.kaggle.com/datasets/sheridanm551/fast-using-lstm-model-steps]( https://www.kaggle.com/datasets/sheridanm551/fast-using-lstm-model-steps).

By using the environment and datasets we've built on *Kaggle*, both models should able to execute by running the `.ipynb` file.

As for other branches, the baseline (opponent) model can be loaded by using the command `torch.load_state_dict(torch.load("./opponent_model/baseline.pt"))`, and all other data files used can be found in `used_datasets` folder.

#### grad/BERT

For this branch, `BERT.ipynb` is for fine tuning the BERT model for Masked Language Modeling (MLM). After the bert model is generated, `Articel_level_grad_BERT.ipynb` can be executed.

#### grad/TF-IDF and TF-IDF/TF-IDF

For those branches, `TFIDF.ipynb` is for generating the list of alternatives by our enhanced TF-IDF method. The pre-generated list of alternatives are stored in `used_dataset/tfidf-words-20.csv`. You can use this to use our model directly.


## Attacking Result

All attacking result is included in each branches for each approaches. It should be a heatmap looks like the picture below:

![phase 1](phase1/picture_result/所有名詞.png)

This is the graphical attacking result for phase 1 attack. Check out files in each folders for more detailed information.

## Contribution

Thanks for the co-authorship from W. -P. Lin, T. -Y. Liu, Z. -W. Hong, H. -L. You, and T. -Y. Hsieh.


