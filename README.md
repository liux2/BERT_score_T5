# T5 with BERTScore
#### Authors: [Xingbang Liu](https://github.com/liux2), [Hualiang Qin](https://github.com/ryanqin)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch-orange)](https://pytorch.org/)

## What's This Project About?

### Brief Intro

BERTScore was proposed by Zhang, et al (2020), as a new metric for generated text
evaluation. Its introduction of contextual embedding not only outperforms common
metrics but also is more robust to the adversarial paraphrases. In the discussion
part of this paper, it is mentioned that BERTScore can be integrated into training
steps for learning loss computation since its differentiability. In this project,
we are going to incorporate BERTScore into the T5 model and train on the WebNLG
2020 dataset to experiment with its performance on Data-to-Text tasks. We also
want to compare the performance of the original T5 model and the modified BERTScore
T5 model to see if the advantage of BERTScore is still preserved as a learning loss.

### Repo Structure

```
.
├── LICENSE
├── README.md
├── datasets
│   ├── dev
│   ├── test
│   └── train
├── scripts
└── src
```

### Dataset

The dataset is [Web NLG 2020 (v3.0)](https://huggingface.co/datasets/web_nlg/tree/main/dummy/release_v3.0_en/0.0.0)

The WebNLG corpus comprises sets of triplets describing facts (entities and
relations between them) and the corresponding facts in the form of natural language
text. The corpus contains sets with up to 7 triplets each along with one or more
reference texts for each set.

## Dependencies

### Data Reader

To read the XML files, WebNLG provides [corpus-reader](https://gitlab.com/webnlg/corpus-reader) to read
the triples and sentences.

### Other Dependences

All the other dependencies were installed in the docker image directly.

## How to Run?

The running environment are encapsulated in the docker image. Follow the steps below:

1. Prepare the repository with the structure in
   [Repo Structure](#repo-structure) section.
2. Build the docker image by running `sudo ./scripts/build_docker.sh` in the
   `BERT_score_T5/` directory.
3. Run the docker image by using
   `sudo ./scripts/run_docker.sh`.

The above steps will create a docker image and run the docker image with
`BERT_score_T5/` repository mounted to the docker volume. To learn how to
customize the Docker image, checkout:

- [How to customize docker](https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html#custcontdockerfile)
- [How to use Nvidia PyTorch NGC Docker images](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

### How to Use Jupyter Notebook with Docker

1. Run `bash scripts/start_jupyter.sh` after logged in docker image.
2. Follow the instructions in terminal and copy paste the link in browser.
    - If you are using local machine, replace `hostname` with `localhost`.
    - If you are using remote machine, use
    `ssh -N -f -L localhost:<remote_port>:localhost:<local_port> <remote_user_name>@<remote_ip>`,
    and don't forget to replace the information in `<>`.

### Why do we use docker?

It saves time. You don't have to worry about installing drivers and dependencies
anymore thanks to Nvidia.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)
file for details.
