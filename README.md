# T5 with BERTScore
#### Authors: [Xingbang Liu](https://github.com/liux2), [Hualiang Qin](https://github.com/ryanqin)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch-orange)](https://pytorch.org/)

## Table of Contents


## What's This Project About?



### Dataset

[Web NLG 2020 (v3.0)](https://huggingface.co/datasets/web_nlg/tree/main/dummy/release_v3.0_en/0.0.0)


## Dependencies


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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)
file for details.
