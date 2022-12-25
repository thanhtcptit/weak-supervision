import os
import sys
import pickle
import warnings
import importlib

import numpy as np

from lstm.DeepLSTM import *
from program_synthesis.synthesizer import Synthesizer
from program_synthesis.heuristic_generator import HeuristicGenerator
warnings.filterwarnings("ignore")


def lsnork_to_l_m(lsnork, num_classes):
    m = 1 - np.equal(lsnork, -1).astype(int)
    l = m * lsnork + (1 - m) * num_classes
    return l, m


def write_txt(name, objs):
    with open(os.path.join(pickle_save, name + '.txt'), 'w') as f:
        for i in objs:
            f.write(i+'\n')


dataset, mode, model, cardinality, num_loop, save_dir, feats = \
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]

loader_file = "data." + dataset + "_loader"
load = importlib.import_module(loader_file)
dl = load.DataLoader()
train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, train_ground, val_ground, test_ground, \
    vizer, val_idx, common_idx, train_text, val_text, test_text = dl.load_data(
        dataset=dataset, split_val=0.1, feat=feats)
print('test length ', len(test_ground))

x = [vizer.get_feature_names()[val_idx[i]] for i in common_idx]

print('Size of validation set ', len(val_ground))
print('Size of train set ', len(train_ground))
print('Size of test set ', len(test_ground))
print('val_primitive_matrix.shape ', val_primitive_matrix.shape)

num_classes = len(np.unique(train_ground))
overall = {}
vals = []

save_path = "LFs/" + dataset + "/" + save_dir
os.makedirs(save_path, exist_ok=True)
print('save_path ', save_path)
val_file_name = mode + '_val_LFs.npy'
train_file_name = mode + '_train_LFs.npy'
test_file_name = mode + '_test_LFs.npy'

if mode == 'all':
    hg = HeuristicGenerator(train_primitive_matrix, val_primitive_matrix, test_primitive_matrix,
                            test_ground, val_ground, train_ground, b=0.5)
    hg.run_synthesizer(max_cardinality=1, idx=None, keep=3, model=model)

    syn = Synthesizer(val_primitive_matrix, val_ground, b=0.5)

    heuristics, feature_inputs = syn.generate_heuristics(model, 1)
    print ("Total Heuristics Generated: ", np.shape(heuristics)[1])
    total_lfs = int(np.shape(heuristics)[1])
    keep_2nd = int(np.ceil(total_lfs / 22))
    keep_1st = int(total_lfs % keep_2nd)
    print('keep_1st, keep_2nd', keep_1st, keep_2nd)
else:
    keep_1st = 3
    keep_2nd = 1

training_marginals = []
HF = []

for j in range(0, 1):
    validation_accuracy = []
    training_accuracy = []
    validation_coverage = []
    training_coverage = []

    idx = None
    if j == 0:
        hg = HeuristicGenerator(train_primitive_matrix, val_primitive_matrix, test_primitive_matrix,
                                test_ground, val_ground, train_ground, b=0.5)
    for i in range(3, num_loop):
        if (i - 2) % 5 == 0:
            print("Running iteration: ", str(i - 2))
        if i == 3:
            hg.run_synthesizer(max_cardinality=cardinality,
                               idx=idx, keep=keep_1st, model=model, mode=mode)
        else:
            hg.run_synthesizer(max_cardinality=cardinality,
                               idx=idx, keep=keep_2nd, model=model, mode=mode)
        hg.run_verifier()

        va, ta, vc, tc, val_lfs, train_lfs, test_lfs, hf = hg.evaluate()
        HF = hf
        validation_accuracy.append(va)
        training_accuracy.append(ta)
        training_marginals.append(hg.vf.train_marginals)
        validation_coverage.append(vc)
        training_coverage.append(tc)

        if i == num_loop - 1:
            np.save(os.path.join(save_path, val_file_name), val_lfs)
            np.save(os.path.join(save_path, train_file_name), train_lfs)
            np.save(os.path.join(save_path, test_file_name), test_lfs)
            print('labels saved')

        hg.find_feedback()
        idx = hg.feedback_idx
        print('Remaining to be labelled ', len(idx))

        if idx == [] and j == 0:
            np.save(os.path.join(save_path, val_file_name), val_lfs)
            np.save(os.path.join(save_path, train_file_name), train_lfs)
            np.save(os.path.join(save_path, test_file_name), test_lfs)
            print('indexes exhausted... now saving labels')
            break

    vals.append(validation_accuracy[-1])
    print("Program Synthesis Train Accuracy: ", training_accuracy[-1])
    print("Program Synthesis Train Coverage: ", training_coverage[-1])
    print("Program Synthesis Validation Accuracy: ", validation_accuracy[-1])


trx = np.load(os.path.join(save_path, train_file_name))
valx = np.load(os.path.join(save_path, val_file_name))
testx = np.load(os.path.join(save_path, test_file_name))

yoyo = list(range(1, num_classes))
yoyo.append(-1)
labels_lfs = []
idxs = []
for i in range(valx.shape[1]):
    for j in yoyo:
        if len(np.where(valx.T[i] == j)[0]) > 1:
            labels_lfs.append(j)
            idxs.append(i)
            break

trx = trx[:, idxs]
testx = testx[:, idxs]
valx = valx[:, idxs]
print(trx.shape, valx.shape, testx.shape)

lx = np.asarray(labels_lfs)
lx[np.where(lx == -1)] = 0
print('LFS are ', lx)
file_name = mode + '_k.npy'
np.save(os.path.join(save_path, file_name), lx)

if feats == 'lstm':
    mkt = MakeTokens()
    train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, vocab_size, embedding_vector_length, \
        max_sentence_length = mkt.make(train_text, val_text, test_text)

upto = int(len(val_ground) / 2)
d_L, U_L = val_ground[:upto], train_ground
d_x, U_x = val_primitive_matrix[:upto], train_primitive_matrix
d_l, U_l = valx[:upto, :], trx

U_text = train_text
d_text = val_text[:upto]
val_text = val_text[upto:]
test_text = test_text

pickle_save = "LFs/" + dataset + "/" + save_dir

write_txt('U', U_text)
write_txt('d', d_text)
write_txt('val', val_text)
write_txt('test', test_text)

d_d = np.array([1.0] * len(d_x))
d_r = np.zeros(d_l.shape)
d_L[np.where(d_L == -1)[0]] = 0

d_l[np.where(d_l == -1)] = 10
d_l[np.where(d_l == 0)] = -1
d_l[np.where(d_l == 10)] = 0
d_l, d_m = lsnork_to_l_m(d_l, num_classes)

file_name = mode + '_d_processed.p'
with open(os.path.join(pickle_save, file_name), "wb") as f:
    pickle.dump(d_x, f)
    pickle.dump(d_l, f)
    pickle.dump(d_m, f)
    pickle.dump(d_L, f)
    pickle.dump(d_d, f)
    pickle.dump(d_r, f)

U_d = np.array([1.0] * len(U_x))
U_r = np.zeros(U_l.shape)

U_L[np.where(U_L == -1)[0]] = 0

U_l[np.where(U_l == -1)] = 10
U_l[np.where(U_l == 0)] = -1
U_l[np.where(U_l == 10)] = 0
U_l, U_m = lsnork_to_l_m(U_l, num_classes)

file_name = mode + '_U_processed.p'
with open(os.path.join(pickle_save, file_name), "wb") as f:
    pickle.dump(U_x, f)
    pickle.dump(U_l, f)
    pickle.dump(U_m, f)
    pickle.dump(U_L, f)
    pickle.dump(U_d, f)
    pickle.dump(U_r, f)


val_L = val_ground[upto:]
val_x = val_primitive_matrix[upto:]
val_l = valx[upto:, :]
val_d = np.array([1.0] * len(val_x))
val_r = np.zeros(val_l.shape)
val_L[np.where(val_L == -1)[0]] = 0

val_l[np.where(val_l == -1)] = 10
val_l[np.where(val_l == 0)] = -1
val_l[np.where(val_l == 10)] = 0

val_l, val_m = lsnork_to_l_m(val_l, num_classes)
file_name = mode + '_validation_processed.p'
with open(os.path.join(pickle_save, file_name), "wb") as f:
    pickle.dump(val_x, f)
    pickle.dump(val_l, f)
    pickle.dump(val_m, f)
    pickle.dump(val_L, f)
    pickle.dump(val_d, f)
    pickle.dump(val_r, f)

test_L = test_ground
test_x = test_primitive_matrix
test_l = testx.copy()
test_d = np.array([1.0] * len(test_x))
test_r = np.zeros(test_l.shape)
test_L[np.where(test_L == -1)[0]] = 0

test_l[np.where(test_l == -1)] = 10
test_l[np.where(test_l == 0)] = -1
test_l[np.where(test_l == 10)] = 0

test_l, test_m = lsnork_to_l_m(test_l, num_classes)
file_name = mode + '_test_processed.p'

with open(os.path.join(pickle_save, file_name), "wb") as f:
    pickle.dump(test_x, f)
    pickle.dump(test_l, f)
    pickle.dump(test_m, f)
    pickle.dump(test_L, f)
    pickle.dump(test_d, f)
    pickle.dump(test_r, f)

print('Final Size of d set ,U set  ,validation set ,test set ',
      len(d_L), len(U_L), len(val_L), len(test_L))

with open(os.path.join(pickle_save, 'generatedLFs.txt'), 'w') as f:
    for j, i in zip(lx, hg.heuristic_stats().iloc[:len(idx)]['Feat 1']):
        f.write(str(j) + ',' + x[int(i)] + '\n')

filepath = './generated_data/' + dataset
file_name = mode + '_reef.npy'
np.save(os.path.join(pickle_save, file_name), training_marginals[-1])
print('final LFs are ', lx.shape)
