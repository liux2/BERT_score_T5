import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor
from bert_score import score
import argparse

# read the argument
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', '-b', type=int, default=8,
                        help='Number of sentence in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=4,
                        help='Number of sweeps over the dataset to train')
parser.add_argument('--types', '-t', type=str, default='f1',
                        help='type of BERTScore to calculate loss (p, r, f1)')
args = parser.parse_args()


# read the Dev and Train file from datasets folder
parent_dir = os.path.dirname(os.getcwd())
dir_path = parent_dir + '/datasets/'
dev_path, train_path = os.path.join(dir_path, 'dev_set.csv'), os.path.join(dir_path, 'train_set.csv')
dev_df, train_df = pd.read_csv(dev_path), pd.read_csv(train_path)

# set the batch size and calculate the number of batches
batch_size = args.batch_size
rows = train_df.shape[0]
train_df = train_df.iloc[:(rows//batch_size*batch_size), :]   # truncated the reminder
train_df = train_df.sample(frac=1)   # shuffle the dataset
num_of_batches = int(len(train_df)/batch_size)

# detect the device
if torch.cuda.is_available():
   dev = torch.device("cuda:0")
   print("Running on the GPU")
else:
   dev = torch.device("cpu")
   print("Running on the CPU")

# set up the model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)
model.to(dev)  # moving the model to GPU

# set up the optimizer
optimizer = Adafactor(model.parameters(),lr=1e-3,
                      eps=(1e-30, 1e-3),
                      clip_threshold=1.0,
                      decay_rate=-0.8,
                      beta1=None,
                      weight_decay=0.0,
                      relative_step=False,
                      scale_parameter=False,
                      warmup_init=False)


# define my own loss function using BERTScore
def my_BERTScore_loss(logits, labels, types="f1"):
    logits = torch.argmax(logits, dim=2)

    row, col = logits.shape
    cands, refs = [], []
    for i in range(row):
        cands.append(tokenizer.decode(logits[i]))
        refs.append(tokenizer.decode(labels[i]))

    p, r, f1 = score(cands, refs, lang="en", verbose=False)
    loss = p if type == 'p' else r if type == 'r' else f1
    loss.requires_grad_(True)
    return -loss.mean()     # set it negative so we can minimize it


# start the training
num_of_epochs = args.epoch
model.train()
loss_per_10_steps = []
for epoch in tqdm(range(1, num_of_epochs + 1)):
    print('Running epoch: {}'.format(epoch))

    running_loss = 0

    for i in tqdm(range(num_of_batches)):
        inputbatch = []
        labelbatch = []
        new_df = train_df[i * batch_size: i * batch_size + batch_size]
        for indx, row in new_df.iterrows():
            input = 'WebNLG: ' + row['triple'] + '</s>'
            labels = row['sentence'] + '</s>'
            inputbatch.append(input)
            labelbatch.append(labels)
        inputbatch = tokenizer.batch_encode_plus(inputbatch, padding=True, max_length=400, return_tensors='pt')[
            "input_ids"]
        labelbatch = tokenizer.batch_encode_plus(labelbatch, padding=True, max_length=400, return_tensors="pt")[
            "input_ids"]
        inputbatch = inputbatch.to(dev)
        labelbatch = labelbatch.to(dev)

        # clear out the gradients of all Variables
        optimizer.zero_grad()

        # Forward propogation
        outputs = model(input_ids=inputbatch, labels=labelbatch)
        logits = outputs.logits
        loss = my_BERTScore_loss(logits, labelbatch, args.types)
        loss_num = loss

        running_loss += loss_num
        if i % 10 == 0:
            loss_per_10_steps.append(loss_num)

        # calculating the gradients
        loss.backward()

        # updating the params
        optimizer.step()

    running_loss = running_loss / int(num_of_batches)
    print('Epoch: {} , Running loss: {}'.format(epoch, running_loss))

torch.save(model.state_dict(), 'BERTScore_T5.bin')
print("finished training and saved the BERTScore model!")
