import torch
from torch.autograd import Variable
from konlpy.tag import Mecab
from torchtext.data import Field, BucketIterator, TabularDataset
from model.bidirec_LSTM import bidirec_LSTM
from collections import defaultdict
import sys

# Argument Setting
model_path_dict = {'-1': './model/self_attn_1H_r5.model',
                   '-2': './model/self_attn_1H_r20.model',
                   '-3': './model/self_attn_3H_r5.model'}

if (len(sys.argv) == 1) or (sys.argv[1] == '-h') or (sys.argv[1] == '-help'):
    print('INSERT ARGUMENTS')
    print('==First==')
    print(' [-1] model1: 1 hidden layer, r=5\n', '[-2] model2: 1 hidden layer, r=5\n', '[-3] model3: 3 hidden layer, r=5')
    print('==Second==')
    print(' [-sample_idx] number from 0 to 781')
    print('==Third==')
    print(' [-(file_path.html)] file path, it is optional, default is "./figures/(file_name)[sample_idx].html"')
    sys.exit()
elif len(sys.argv) == 2:
    print('Please Insert 2nd Argument: [-sample_idx] number from 0 to 781')
    sys.exit()
elif len(sys.argv) == 3:
    model_path = model_path_dict[sys.argv[1]]
    sample_idx = int(sys.argv[2].strip('-'))
    vis_path = None
elif len(sys.argv) == 4:
    model_path = model_path_dict[sys.argv[1]]
    sample_idx = int(sys.argv[2].strip('-'))
    vis_path = sys.argv.strip('-')
else:
    sys.exit()

USE_CUDA = torch.cuda.is_available()
DEVICE = 0 if USE_CUDA else -1

batch_size = 64

# Tokenizer
tagger = Mecab()
tagger = tagger.morphs

# Make Field
REVIEW = Field(tokenize=tagger, use_vocab=True, lower=True, #init_token="<s>", eos_token="</s>",
               include_lengths=True, batch_first=True)
LABEL = Field(sequential=False, use_vocab=False, preprocessing=lambda x: int(x))

# Get train/test data
train_data, test_data = TabularDataset.splits(
                   path="./data/", train='train_docs.txt', validation="test_docs.txt",
                   format='tsv', fields=[('review', REVIEW), ('label', LABEL)])

# Build Vocaburary
REVIEW.build_vocab(train_data)
len(REVIEW.vocab)

# Make iterator for splits
train_iter, test_iter = BucketIterator.splits(
    (train_data, test_data), batch_size=batch_size, device=DEVICE, # device -1 : cpu, device 0 : 남는 gpu
    sort_key=lambda x: len(x.review), sort_within_batch=True, repeat=False) # x.TEXT 길이 기준으로 정렬

# parameters
V = len(REVIEW.vocab)
D = 100
H = 200
H_f = 1000
O = 1
da = 300
num_directions = 2
bidirec = True
batch_size = 64
LR = 0.01
STEP = 10
num_layers = int(model_path.split('_')[2][0])
r = int(model_path.split('_')[3].split('.')[0][1:])

# Load model
model = bidirec_LSTM(V, D, H, H_f, O, da, r, num_layers=num_layers, bidirec=bidirec, use_cuda=USE_CUDA)
if USE_CUDA:
    model = model.cuda()
    model.load_state_dict(torch.load(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))


def save_data(i, model, inputs, lengths, targets, preds, dic):
    dic[i] = {'inputs': inputs, 'lengths': lengths, 'A': model.A, 'targets': targets, 'preds': preds}
    return dic


num_equal = 0
dic = defaultdict(dict)
for i, batch in enumerate(test_iter):
    inputs, lengths = batch.review
    targets = batch.label
    if 0 in lengths:
        idxes = torch.arange(inputs.size(0))
        if USE_CUDA:
            idxes = idxes.cuda()
        mask = Variable(idxes[lengths.ne(0)].long())

        inputs = inputs.index_select(0, mask)
        lengths = lengths.masked_select(lengths.ne(0))
        targets = targets.index_select(0, mask)

    preds = model.predict(inputs, lengths)
    num_equal += torch.eq(preds, targets).sum().data[0]
    dic = save_data(i, model, inputs, lengths, targets, preds, dic)
acc = num_equal / len(test_data)


def get_reviews(inputs, lengths):
    max_len = max(lengths)
    reviews = []
    for s, l in zip(inputs, lengths):
        if l == max_len:
            reviews.append([REVIEW.vocab.itos[w.data[0]] for w in s])
        else:
            num_add_pad = max_len - l
            reviews.append([REVIEW.vocab.itos[w.data[0]] for w in s] + ['<pad>']*(num_add_pad-1))
    return reviews


def build_matrix(dic, sample_idx):
    A = dic[sample_idx]['A'].sum(1)
    A_norm = (A / A.max(1)[0].unsqueeze(1)).data.numpy()
    inputs = dic[sample_idx]['inputs']
    lengths = dic[sample_idx]['lengths']
    targets = dic[sample_idx]['targets']
    preds = dic[sample_idx]['preds']

    reviews = get_reviews(inputs, lengths)
    return reviews, A_norm, targets, preds


def span_str(color, input_str):
    string = '<span style="margin-left:3px;background-color:rgba({})">{}</span>'.format(color, input_str)
    return string


def write_html_vis(reviews, A_norm, targets, preds, vis_path, model_path, acc):
    with open(vis_path, 'w', encoding='utf-8') as file:
        print('<html style="margin:0;padding:0;"><body style="margin:0;padding:0;">\n', file=file)
        print('<div style="margin:25px;">\n', file=file)
        print('<h1>Model: {} | Total Accuracy: {:.4f}</h1>'.format(model_path.split('/')[-1].split('.')[0], acc), file=file)
        print('<h2> This Sample Accurary: {:.4f}</h2>'.format((torch.eq(targets, preds).float().sum()/targets.size(0)).data[0]), file=file)
        for review, score, target, preds in zip(reviews, A_norm, targets.data.numpy(), preds.data.numpy()):
            alphas = ['{:.2f}'.format(s) for s in score]
            print('<p style="margin:10px;">\n', file=file)
            if target == preds:
                color = '154,226,143, 0.7'
                input_str = '[target: {}| pred: {}]'
                string = span_str(color, input_str)
                print(string.format(target, preds), file=file)
            else:
                color = '203,127,230, 0.7'
                input_str = '[target: {}| pred: {}]'
                string = span_str(color, input_str)
                print(string.format(target, preds), file=file)

            for word, alpha in zip(review, alphas):
                color = '255,50,50, {}'
                input_str = '{}'
                string = span_str(color, input_str)
                print(('\t' + string + '\n').format(alpha, word), file=file)
            print('</p>\n', file=file)
        print('</div>\n', file=file)
        print('</body></html>', file=file)
if not vis_path:
    vis_path = './figures/{}[{}].html'.format(model_path.split('/')[-1].split('.')[0], sample_idx)
reviews, A_norm, targets, preds = build_matrix(dic, sample_idx)
write_html_vis(reviews, A_norm, targets, preds, vis_path, model_path, acc)
print('Done! File is in {}'.format(vis_path))