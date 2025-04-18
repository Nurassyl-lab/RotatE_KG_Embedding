#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

import sys

from sklearn.decomposition import TruncatedSVD
from scipy.stats import pearsonr
from sklearn.kernel_approximation import RBFSampler

def rbf_feature_map(emb: np.ndarray, gamma: float, n_components: int):
    """
    Approximate RBF kernel φ(x) via Random Fourier Features.
    Returns φ(X) and the fitted RBFSampler.
    """
    sampler = RBFSampler(gamma=gamma, n_components=n_components, random_state=42)
    feats = sampler.fit_transform(emb)
    return feats, sampler

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.train_positive_loss = []
        self.train_negative_loss = []
        self.train_loss = []
        self.val_mrr = []
        self.val_mr = []
        self.val_hit1 = []
        self.val_hit3 = []
        self.val_hit10 = []

        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)


        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score
    
    def freeze_old_embeddings_hook(grad, n_old):
        # Create a mask with zeros for old embeddings and ones for new embeddings
        mask = torch.ones_like(grad)
        mask[:n_old] = 0  # Freeze pre-trained embeddings
        return grad * mask
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args, n_old=None):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        # freeze old embeddings by
        # zero gradients for the pre-trained embeddings (first n_old rows)
        if args.mask_old_embeddings:
            if model.entity_embedding.grad is not None:
                model.entity_embedding.grad[:n_old].zero_()

        # train using compression SVD
        if args.do_svd:
            model = KGEModel.svd_emd(model, 50)

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }
        model.train_positive_loss.append(positive_sample_loss.item())
        model.train_negative_loss.append(negative_sample_loss.item())
        model.train_loss.append(loss.item())

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)
                if metric == "MRR":
                    model.val_mrr.append(metrics[metric])
                elif metric == "MR":
                    model.val_mr.append(metrics[metric])
                elif metric == "HITS@1":
                    model.val_hit1.append(metrics[metric])
                elif metric == "HITS@3":
                    model.val_hit3.append(metrics[metric])
                elif metric == "HITS@10":
                    model.val_hit10.append(metrics[metric])

            # save the losses and metrics into a numpy array
            # save metrics
            np.save(f"{args.save_path}/val_mrr.npy", np.array(model.val_mrr))
            np.save(f"{args.save_path}/val_mr.npy", np.array(model.val_mr))
            np.save(f"{args.save_path}/val_hit1.npy", np.array(model.val_hit1))
            np.save(f"{args.save_path}/val_hit3.npy", np.array(model.val_hit3))
            np.save(f"{args.save_path}/val_hit10.npy", np.array(model.val_hit10))

            # save losses
            np.save(f"{args.save_path}/train_positive_loss.npy", np.array(model.train_positive_loss))
            np.save(f"{args.save_path}/train_negative_loss.npy", np.array(model.train_negative_loss))
            np.save(f"{args.save_path}/train_loss.npy", np.array(model.train_loss))
        
        return metrics

    @staticmethod
    def generate_embeddings(model, args):
        """
        Generate and return entity and relation embeddings from the model.
        """
        model.eval()
        with torch.no_grad():
            # Get the embeddings (they are nn.Parameters, so no .weight attribute)
            entity_embeddings = model.entity_embedding.detach()
            relation_embeddings = model.relation_embedding.detach()
            
            # Move to CPU if using CUDA
            if args.cuda:
                entity_embeddings = entity_embeddings.cpu()
                relation_embeddings = relation_embeddings.cpu()
            
            # Convert tensors to numpy arrays
            entity_embeddings = entity_embeddings.numpy()
            relation_embeddings = relation_embeddings.numpy()
        
        print(f"Entity embeddings shape: {entity_embeddings.shape}")
        print(f"Relation embeddings shape: {relation_embeddings.shape}")

        return entity_embeddings, relation_embeddings

    @staticmethod
    def normalize_entity_emd(model):
        entity_embeddings = model.entity_embedding.detach().clone()

        # Split into real and imaginary parts
        num_entities, embedding_dim = entity_embeddings.shape
        half_dim = embedding_dim // 2  # Since first half is real, second half is imaginary

        real_part = entity_embeddings[:, :half_dim]
        imag_part = entity_embeddings[:, half_dim:]

        # Compute complex magnitude
        norm = torch.sqrt(real_part**2 + imag_part**2 + 1e-10)  # Add small epsilon to avoid division by zero

        # Normalize real and imaginary parts
        real_part = real_part / norm
        imag_part = imag_part / norm

        # Recombine into a single tensor
        normalized_embeddings = torch.cat([real_part, imag_part], dim=1)

        # Update back into the model
        model.entity_embedding.data = normalized_embeddings
        return model
    
    @staticmethod
    def svd_and_normalize_entity_emd(model, reduced_dim):
        """
        Apply SVD to reduce the dimensions of the entity embeddings and normalize the result.

        Args:
            model: The model containing the entity embeddings to be reduced.
            reduced_dim: The target dimensionality after SVD reduction.

        Returns:
            model: The model with normalized and reduced embeddings.
        """
        pi = 3.14159265358979323846
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Detach embeddings and convert to numpy array for SVD
        entity_embeddings = model.entity_embedding.detach().cpu().numpy()
        relation_embeddings = model.relation_embedding.detach().cpu().numpy()

        # scale from -pi to pi
        entity_embeddings = entity_embeddings/(model.embedding_range.item()/pi)
        relation_embeddings = relation_embeddings/(model.embedding_range.item()/pi)

        # Perform SVD for dimensionality reduction
        svd_entity = TruncatedSVD(n_components=reduced_dim, random_state=42)
        svd_relation = TruncatedSVD(n_components=reduced_dim, random_state=42)
        reduced_embeddings_entity = svd_entity.fit_transform(entity_embeddings)
        reduced_embeddings_relation = svd_relation.fit_transform(relation_embeddings)

        # Convert reduced embeddings back to a torch tensor
        entity_embeddings = torch.tensor(reduced_embeddings_entity, dtype=torch.float32)
        relation_embeddings = torch.tensor(reduced_embeddings_relation, dtype=torch.float32)

        # * (Optional) Reverse normalization from -pi to pi back to original scale
        # entity_embeddings = entity_embeddings * (model.embedding_range.item() / pi)
        # relation_embeddings = relation_embeddings * (model.embedding_range.item() / pi)

        # * (Optional) Normalize the reduced embeddings (L2 normalization)
        # norm = torch.norm(entity_embeddings, p=2, dim=1, keepdim=True)
        # entity_embeddings = entity_embeddings / (norm + 1e-10)  # Add small epsilon to avoid division by zero
        # norm = torch.norm(relation_embeddings, p=2, dim=1, keepdim=True)
        # relation_embeddings = relation_embeddings / (norm + 1e-10)  # Add small epsilon to avoid division by zero

        # Update the model with the normalized embeddings
        model.entity_embedding.data = entity_embeddings.to(device)
        model.relation_embedding.data = relation_embeddings.to(device)
        return model
    
    @staticmethod
    def svd_emd(model, reduced_dim):
        """
        Apply SVD to reduce the dimensions of the entity embeddings and project back to the original size.

        Args:
            model: The model containing the entity embeddings to be reduced.
            reduced_dim: The target dimensionality after SVD reduction.

        Returns:
            model: The model with embeddings projected back to the original size.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        entity_embeddings = model.entity_embedding.detach().cpu().numpy()
        relation_embeddings = model.relation_embedding.detach().cpu().numpy()

        # Perform SVD for dimensionality reduction
        U_entity, S_entity, Vt_entity = np.linalg.svd(entity_embeddings, full_matrices=False)
        U_relation, S_relation, Vt_relation = np.linalg.svd(relation_embeddings, full_matrices=False)

        # Reduce dimensions
        reduced_entity = np.dot(U_entity[:, :reduced_dim], np.diag(S_entity[:reduced_dim]))
        reduced_relation = np.dot(U_relation[:, :reduced_dim], np.diag(S_relation[:reduced_dim]))

        # Project back to the original size
        projected_entity = np.dot(reduced_entity, Vt_entity[:reduced_dim, :])
        projected_relation = np.dot(reduced_relation, Vt_relation[:reduced_dim, :])

        # Convert back to torch tensors
        model.entity_embedding.data = torch.tensor(projected_entity, dtype=torch.float32).to(device)
        model.relation_embedding.data = torch.tensor(projected_relation, dtype=torch.float32).to(device)

        return model
