from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import accuracy, masked_loss, masked_acc
from pygcn.models import GCN
from pygcn.data import load_data, preprocess_features, preprocess_adj

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
                    help='Dataset string')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset_str=args.dataset)
logger.info('adj: {}'.format(adj.shape))
logger.info('features: {}'.format(features.shape))
logger.info('y tr{} val{} te{} '.format(y_train.shape, y_val.shape, y_test.shape))
logger.info('mask tr{} val{} te{}'.format(train_mask.shape, val_mask.shape, test_mask.shape))

features = preprocess_features(features) # [49216, 2], [49216], [2708, 1433]
supports = preprocess_adj(adj)

device = torch.device('cuda')
train_label = torch.from_numpy(y_train).long().to(device)
num_classes = train_label.shape[1]
train_label = train_label.argmax(dim=1)
train_mask = torch.from_numpy(train_mask.astype(np.int)).float().to(device)
val_label = torch.from_numpy(y_val).long().to(device)
val_label = val_label.argmax(dim=1)
val_mask = torch.from_numpy(val_mask.astype(np.int)).to(device)
test_label = torch.from_numpy(y_test).long().to(device)
test_label = test_label.argmax(dim=1)
test_mask = torch.from_numpy(test_mask.astype(np.int)).to(device)

i = torch.from_numpy(features[0]).long().to(device)
v = torch.from_numpy(features[1]).to(device)
feature = torch.sparse.FloatTensor(i.t(), v, features[2]).float().to(device)

i = torch.from_numpy(supports[0]).long().to(device)
v = torch.from_numpy(supports[1]).to(device)
support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float().to(device)

logger.info('x :{}'.format(feature.shape))
logger.info('sp: {}'.format(support.shape))
num_features_nonzero = feature._nnz()
feat_dim = feature.shape[1]

# Model and optimizer
model = GCN(nfeat=feat_dim,
            nhid=args.hidden,
            nclass=num_classes,
            dropout=args.dropout)
model.to(device)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
model.train()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    out = model(feature, support)
    logger.info ("out: {}".format(out.shape))
    logger.info ("train_label {}".format(train_label.shape))
    logger.info ("train_mask {}".format(train_mask.shape))
    # out = output[0]
    loss_train = masked_loss(out, train_label, train_mask)
    loss_train += args.weight_decay * model.l2_loss()

    acc_train = masked_acc(out, train_label, train_mask)
    loss_train.backward()
    optimizer.step()

    loss_val = masked_loss(out, val_label, val_mask)
    acc_val = masked_acc(out, val_label, val_mask)
    logger.info('Epoch: {:04d}, loss_train: {:.4f}, acc_train: {:.4f}, loss_val: {:.4f}, acc_val: {:.4f}, time: {:.4f}s\
    '.format(epoch+1, loss_train.item(), acc_train.item(), loss_val.item(), acc_val.item(), time.time() - t))


def test():
    model.eval()
    out = model(feature, support)
    # out = out[0]
    loss_test = masked_loss(out, test_label, test_mask)
    acc_test = masked_acc(out, test_label, test_mask)
    logger.info("Test set results: \
        loss= {:.4f}, accuracy= {:.4f}".format(loss_test.item(), acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
logger.info("Optimization Finished!")
logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()