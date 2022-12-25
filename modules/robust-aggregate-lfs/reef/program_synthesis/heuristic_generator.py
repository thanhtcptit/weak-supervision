import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import random
from program_synthesis.synthesizer import Synthesizer
from program_synthesis.verifier import Verifier

class HeuristicGenerator(object):
    def __init__(self, train_primitive_matrix, val_primitive_matrix, test_primitive_matrix,
                 test_ground, val_ground, train_ground=None, b=0.5):
        self.train_primitive_matrix = train_primitive_matrix
        self.val_primitive_matrix = val_primitive_matrix
        self.val_ground = val_ground
        self.train_ground = train_ground
        self.test_primitive_matrix = test_primitive_matrix
        self.test_ground = test_ground
        
        self.num_classes = len(np.unique(self.train_ground))
        self.b = 1 / (self.num_classes)
        print('self.b ', self.b)

        self.vf = None
        self.syn = None
        self.hf = []
        self.feat_combos = []
        self.val_lfs = []
        self.train_lfs = []
        self.all_idx = set()

    def apply_heuristics(self, heuristics, primitive_matrix, feat_combos, beta_opt):
        def marginals_to_labels(hf,X,beta):
            prob_classes = hf.predict_proba(X)
            marginals = np.max(prob_classes, axis=1)
            labels_cutoff = np.argmax(prob_classes, axis=1)
            labels_cutoff[labels_cutoff == 0] = -1
            labels_cutoff[np.logical_and((self.b -beta) <= marginals, marginals <= (self.b + beta))] = 0
            return labels_cutoff

        L = np.zeros((np.shape(primitive_matrix)[0],len(heuristics)))
        try:
            uu = []
            for i in feat_combos:
                uu.append(i[0])
        except:
            f = feat_combos
            feat_combos = []
            feat_combos.append(f)
        for i,hf in enumerate(heuristics):
            if type(feat_combos[i]) is not tuple:
                feat_combos[i] = (feat_combos[i],)
            L[:, i] = marginals_to_labels(hf,primitive_matrix[:, feat_combos[i]], beta_opt[i])

        self.labels = L
        return L

    def prune_heuristics(self,heuristics,feat_combos,keep=1, mode='normal'):
        def calculate_jaccard_distance(num_labeled_total, num_labeled_L):
            scores = np.zeros(np.shape(num_labeled_L)[1])
            for i in range(np.shape(num_labeled_L)[1]):
                scores[i] = np.sum(np.minimum(num_labeled_L[:, i], num_labeled_total)) \
                    / np.sum(np.maximum(num_labeled_L[:, i], num_labeled_total))
            return 1 - scores
        
        L_val = np.array([])
        L_train = np.array([])
        beta_opt = np.array([])
        max_cardinality = len(heuristics)
        for i in range(max_cardinality):
            beta_opt_temp = self.syn.find_optimal_beta(
                heuristics[i], self.val_primitive_matrix, feat_combos[i], self.val_ground)
            L_temp_val = self.apply_heuristics(
                heuristics[i], self.val_primitive_matrix, feat_combos[i], beta_opt_temp) 
            self.val_lfs = L_temp_val
            L_temp_train = self.apply_heuristics(
                heuristics[i], self.train_primitive_matrix, feat_combos[i], beta_opt_temp) 
            self.train_lfs = L_temp_train
            L_temp_test = self.apply_heuristics(
                heuristics[i], self.test_primitive_matrix, feat_combos[i], beta_opt_temp) 
            self.test_lfs = L_temp_test
            
            beta_opt = np.append(beta_opt, beta_opt_temp)
            if i == 0:
                L_val = np.append(L_val, L_temp_val)
                L_val = np.reshape(L_val,np.shape(L_temp_val))
                L_train = np.append(L_train, L_temp_train)
                L_train = np.reshape(L_train,np.shape(L_temp_train))
            else:
                L_val = np.concatenate((L_val, L_temp_val), axis=1)
                L_train = np.concatenate((L_train, L_temp_train), axis=1)
        
        acc_cov_scores = [f1_score(self.val_ground, L_val[:,i], average='weighted')
                          for i in range(np.shape(L_val)[1])] 
        acc_cov_scores = np.nan_to_num(acc_cov_scores)

        if self.vf != None:
            train_num_labeled = np.sum(np.abs(self.vf.L_train.T), axis=0) 
            jaccard_scores = calculate_jaccard_distance(train_num_labeled,np.abs(L_train))
        else:
            jaccard_scores = np.ones(np.shape(acc_cov_scores))

        combined_scores = 0.5*acc_cov_scores + 0.5*jaccard_scores
        if mode == 'random':
            tmp = np.argsort(combined_scores)[::-1]
            sort_idx = random.sample(range(0,len(tmp)), keep)
        else:
            sort_idx = np.argsort(combined_scores)[::-1][0:keep]
        
        return sort_idx
     

    def run_synthesizer(self, max_cardinality=1, idx=None, keep=1, model='lr', mode='normal'):
        if idx == None:
            primitive_matrix = self.val_primitive_matrix
            ground = self.val_ground
        else:
            primitive_matrix = self.val_primitive_matrix[idx,:]
            ground = self.val_ground[idx]

        self.syn = Synthesizer(primitive_matrix, ground, b=self.b)

        def index(a, inp):
            i = 0
            remainder = 0
            while inp >= 0:
                remainder = inp
                inp -= len(a[i])
                i += 1
            try:
                return a[i - 1][remainder]
            except:
                import pdb; pdb.set_trace()

        hf, feat_combos = self.syn.generate_heuristics(model, max_cardinality)
        for m in range(max_cardinality):
            if len(self.all_idx) > 0:
                self.all_idx = list(self.all_idx)
                
                tmp = np.asarray(self.all_idx)
                rmv = np.where(tmp < len(hf[m]))
                h = np.delete(hf[m],tmp[rmv])
                hf[m] = []
                for i in h:
                    hf[m].append(i)

                if len(tmp[rmv]) > 0:
                    t = np.sort(tmp[rmv])[::-1]
                    for i in t:
                        del feat_combos[m][i]

        sort_idx = self.prune_heuristics(hf,feat_combos, keep, mode)
        self.all_idx = set(self.all_idx)
        for i in sort_idx:
            self.all_idx.add(i)
        for i in sort_idx:
            self.hf.append(index(hf,i)) 
            self.feat_combos.append(index(feat_combos,i))

        beta_opt = self.syn.find_optimal_beta(self.hf, self.val_primitive_matrix, self.feat_combos, self.val_ground)
        self.L_val = self.apply_heuristics(self.hf, self.val_primitive_matrix, self.feat_combos, beta_opt)       
        self.L_train = self.apply_heuristics(self.hf, self.train_primitive_matrix, self.feat_combos, beta_opt)  
        self.L_test = self.apply_heuristics(self.hf, self.test_primitive_matrix, self.feat_combos, beta_opt)  
    
    def run_verifier(self):
        self.vf = Verifier(self.L_train, self.L_val, self.val_ground, has_snorkel=False)
        self.vf.train_gen_model()
        self.vf.assign_marginals()

    def gamma_optimizer(self, marginals):
        m = len(self.hf)
        gamma = 0.5-(1/(m**(3/2.))) 
        return gamma

    def find_feedback(self):
        gamma_opt = self.gamma_optimizer(self.vf.val_marginals)
        vague_idx = self.vf.find_vague_points(b=self.b, gamma=gamma_opt)
        incorrect_idx = vague_idx
        self.feedback_idx = list(set(list(np.concatenate((vague_idx,incorrect_idx)))))   


    def evaluate(self):
        self.val_marginals = self.vf.val_marginals
        self.train_marginals = self.vf.train_marginals

        def calculate_accuracy(marginals, b, ground):
            total = np.shape(np.where(marginals != 0.5))[1]
            labels = np.sign(2*(marginals - 0.5))
            return np.sum(labels == ground)/float(total)
    
        def calculate_coverage(marginals, b, ground):
            total = np.shape(np.where(marginals != 0.5))[1]
            labels = np.sign(2*(marginals - 0.5))
            return total/float(len(labels))

        self.val_accuracy = calculate_accuracy(self.val_marginals, self.b, self.val_ground)
        self.train_accuracy = calculate_accuracy(self.train_marginals, self.b, self.train_ground)
        self.val_coverage = calculate_coverage(self.val_marginals, self.b, self.val_ground)
        self.train_coverage = calculate_coverage(self.train_marginals, self.b, self.train_ground)
        return self.val_accuracy, self.train_accuracy, self.val_coverage, self.train_coverage , self.L_val, self.L_train, self.L_test, self.hf

    def heuristic_stats(self):
        def calculate_accuracy(marginals, b, ground):
            total = np.shape(np.where(marginals != 0.5))[1]
            labels = np.sign(2*(marginals - 0.5))
            return np.sum(labels == ground)/float(total)
    
        def calculate_coverage(marginals, b, ground):
            total = np.shape(np.where(marginals != 0))[1]
            labels = marginals
            return total/float(len(labels))

        stats_table = np.zeros((len(self.hf),7))
        for i in range(len(self.hf)):
            stats_table[i,0] = int(self.feat_combos[i][0])
            try:
                stats_table[i,1] = int(self.feat_combos[i][1])
            except:
                stats_table[i,1] = -1.
            try:
                stats_table[i,2] = int(self.feat_combos[i][2])
            except:
                stats_table[i,2] = -1.    
            
            stats_table[i,3] = calculate_accuracy(self.L_val[:,i], self.b, self.val_ground)
            stats_table[i,4] = calculate_accuracy(self.L_train[:,i], self.b, self.train_ground)
            stats_table[i,5] = calculate_coverage(self.L_val[:,i], self.b, self.val_ground)
            stats_table[i,6] = calculate_coverage(self.L_train[:,i], self.b, self.train_ground)
        
        column_headers = ['Feat 1', 'Feat 2','Feat 3', 'Val Acc', 'Train Acc', 'Val Cov', 'Train Cov']
        pandas_stats_table = pd.DataFrame(stats_table, columns=column_headers)
        return pandas_stats_table


            


