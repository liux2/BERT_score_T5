FROM nvcr.io/nvidia/pytorch:22.03-py3

# Install my-extra-package-1 and my-extra-package-2
#RUN apt-get update && apt-get install -y --no-install-recommends \
#	my-extra-package-1 \
#	my-extra-package-2 \
#	&& \
#rm -rf /var/lib/apt/lists/
RUN pip install git+https://github.com/huggingface/transformers
RUN pip install datasets
RUN pip install sentencepiece
RUN pip install bert-score
