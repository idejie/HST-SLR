import pandas as pd
import torch
from transformers import CLIPTextModel, CLIPTokenizer

target = 'phoenix2014' # phoenix2014, phoenix2014T, CSLDaily

df = pd.read_csv('./sentence_cluster/description_{}.csv'.format(target))

'''
clip_path = "./pretrained/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(clip_path, local_files_only=True, torch_dtype=torch.float16)
text_encoder = CLIPTextModel.from_pretrained(clip_path, local_files_only=True, torch_dtype=torch.float16).to('cuda')
'''
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

l1 = df['l1_label'].max() + 1
l1_list = []
for i in range(l1):
    print('process node' + str(i) + ' in l1')
    text_inputs = df[df['l1_label'] == i]['sentence'].tolist()
    token = tokenizer(text_inputs, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    nd = text_encoder(token.input_ids.to("cuda"))[0].half() # m x 77 x 768
    nd = torch.mean(nd, dim=0, keepdim=True) # 1 x 77 x 768
    nd = nd.detach()
    l1_list.append(nd)

l1_prototype = torch.cat(l1_list, dim=0).reshape(l1, -1) # l1 x 77 x 768
torch.save(l1_prototype, "./HDT_prototype/l1_{}.pt".format(target))

l2_max = df['l2_label'].max() + 1
padded_tf_list = []
for i in range(l1):
    print('process child nodes of node' + str(i) + ' in l1')
    subset_df = df[df['l1_label'] == i]

    l2 = subset_df['l2_label'].max() + 1
    l2_list = []
    for j in range(l2):
        text_inputs = subset_df[subset_df['l2_label'] == j]['sentence'].tolist()
        token = tokenizer(text_inputs, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        nd = text_encoder(token.input_ids.to("cuda"))[0].half() # m x 77 x 768
        nd = torch.mean(nd, dim=0, keepdim=True) # 1 x 77 x 768
        nd = nd.detach()
        l2_list.append(nd)

    text_feature = torch.cat(l2_list, dim=0) # l2 x 77 x 768

    padded_tf = torch.nn.functional.pad(text_feature, (0, 0, 0, 0, 0, l2_max - l2)) # l2_max x 77 x 768
    padded_tf_list.append(padded_tf)

l2_prototype = torch.cat(padded_tf_list, dim=0).view(l1, l2_max, -1) # l1 x l2_max x 77 x 768
l2_prototype = l2_prototype.detach()
torch.save(l2_prototype, "./HDT_prototype/l2_{}.pt".format(target))
