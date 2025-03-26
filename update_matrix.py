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

alpha = 1.5
l1 = df['l1_label'].max() + 1
l2_max = df['l2_label'].max() + 1

all_list = []
for i in range(l1):
    subset_df = df[df['l1_label'] == i]
    up_row = torch.ones(1, num_classes)
    for idx in subset_df['word_index']:
        up_row[0, idx+1] = alpha

    l2 = subset_df['l2_label'].max() + 1
    up_list = []
    for j in range(l2):
        min_df = subset_df[subset_df['l2_label'] == j]
        new_up_row = up_row.clone()
        for idx in min_df['word_index']:
            new_up_row[0, idx+1] = alpha * alpha
        up_list.append(new_up_row)

    up_row = torch.cat(up_list, dim=0) # l2 x n
    if l2 < l2_max:
        temp = torch.ones(l2_max-l2, num_classes)
        up_row = torch.cat((up_row, temp), dim=0) # l2_max x n

    all_list.append(up_row)

up_matrix = torch.cat(all_list, dim=0).view(l1, l2_max, num_classes) # l1 x l2_max x n
torch.save(up_matrix, "./HDT_prototype/up_matrix_{}.pt".format(target))
