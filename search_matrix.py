import pandas as pd
import torch
import argparse

parser = argparse.ArgumentParser(description='train for SLG')
parser.add_argument('--dataset', default="phoenix2014", help='the target dataset')
args = parser.parse_args()

target = args.dataset # phoenix2014, phoenix2014T, CSLDaily

df = pd.read_csv('./sentence_cluster/description_{}.csv'.format(target))

value_mapping = {
    "phoenix2014": 1296,
    "phoenix2014T": 1116,
    "CSLDaily": 2001
}
num_classes = value_mapping.get(target, -1)

if num_classes == -1:
    print("target error")
    exit(1)

l1 = df['l1_label'].max() + 1

ls_list = []
for i in range(l1):
    subset_df = df[df['l1_label'] == i]
    ls_row = torch.zeros((1, num_classes))
    for idx in subset_df['word_index']:
        ls_row[0, idx+1] = 1
    ls_list.append(ls_row)

ls_matrix = torch.cat(ls_list, dim=0).view(num_classes, -1) # n x l1
sums = ls_matrix.sum(dim=1, keepdim=True)
sums[sums == 0] = 1
ls_matrix = ls_matrix / sums
torch.save(ls_matrix, "./HDT_prototype/ls_matrix_{}.pt".format(target))
