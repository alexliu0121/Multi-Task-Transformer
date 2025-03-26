def main():
    import pandas as pd
    import torch
    import torch.nn as nn
    import time
    from tqdm.auto import tqdm
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence
    from sklearn.preprocessing import MultiLabelBinarizer
    from typing import List
    import re
    import string
    import math
    import os
    import numpy as np
    from collections import Counter
    from torch.utils.data import Dataset, DataLoader
    import torch.optim as optim
    import torch.nn.functional as F
    import sklearn
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.metrics import classification_report as  classification_report_rel
    from seqeval.scheme import IOB2
    from seqeval.metrics import classification_report as classification_report_tag
    from tqdm import tqdm
    import spacy
    from sklearn.model_selection import train_test_split
    import torch
    from gensim.models import Word2Vec
    import gensim
    from itertools import chain
    from argparse import ArgumentParser
    import pickle

    parser = ArgumentParser("example")
    parser.add_argument('--train', action="store_true", help="indicator to train model")
    parser.add_argument('--test', action="store_true", help="indicator to test model")
    parser.add_argument('--data', help="path to data file")
    parser.add_argument('--save_model', help="ouput path of trained model")
    parser.add_argument('--model_path', help="path to load trained model from")
    parser.add_argument('--output', help="output path of predictions")
    args = parser.parse_args()

    parameters = {
        "num_class": 2,
        "learning_rate": 6e-4, # the speed that model learn
        "epochs": 30, # If U would fine-tune it, the epochs didn't need to set too much
        "batch_size": 8, 
        "dropout": 0.1, # how random amount will be give up
        "hidden_dim": 32,
        "optimizer": 'Adam',
        "alpha":0.5, # 1 for only train tag, 0 for only train relations, 0.5 for both
        "d_model":300,
        "n_heads":3,
        "max_len":32,
        "pad_token_id":0,
        "num_encoder_layers":3,
    }

    if args.train:
        # train_model(args.data, args.save_model)
        device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

        path_to_glove_embedding = './glove.6B.300d.txt'
        path_to_glove_embeddings_in_gensim_word2vec_format = './glove-word2vec.6B.300d.txt'
        from gensim.scripts.glove2word2vec import glove2word2vec
        from gensim.test.utils import datapath, get_tmpfile
        glove_file = datapath(path_to_glove_embedding)
        tmp_file = get_tmpfile('./glove-word2vec.6B.300d.txt')
        _ = glove2word2vec(glove_file, tmp_file)
        # # Load glove6B embeddings into Gensim class containing matrix
        word2vec_weights = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)

        # Hint: input data, change the file path here
        train_data = pd.read_csv(args.data)

        # Hint: do some preprocessing
        #shuffle data
        train_data=train_data.sample(frac = 1,random_state=42)
        train_data=train_data.fillna('none')
        df=train_data

        def tokenize(s):
            return s.split()
        
        df['utterances'] = df['utterances'].apply(tokenize)
        df['IOB Slot tags'] = df['IOB Slot tags'].apply(lambda x: x.replace('_', '-')).apply(tokenize)
        df['Core Relations'] = df['Core Relations'].fillna("").apply(tokenize)

        vocab = {}
        vocab['<pad>'] = 0
        vocab['<unk>'] = 1
        for idx, token in enumerate(df['utterances'].explode().unique(), start=2):
            vocab[token] = idx
        idx2token = {v:k for k,v in vocab.items()}

        idx2tag = dict(enumerate(df['IOB Slot tags'].explode().unique()))
        tag2idx = {v:k for k,v in idx2tag.items()}

        idx2rel = dict(enumerate(df['Core Relations'].explode().dropna().unique()))
        rel2idx = {v:k for k,v in idx2rel.items()}

        mlb = MultiLabelBinarizer(classes=list(rel2idx.keys()))
        mlb.fit(df['Core Relations'])
        df['Core Relations'] = mlb.transform(df['Core Relations']).tolist()

        idx2tag
        labels_tag=[]
        labels_rel=[]
        for tags in idx2tag.items():
            labels_tag.append(tags[1])
        for rels in idx2rel.items():
            labels_rel.append(rels[1])

        class JointTrainingDataset(Dataset):
            def __init__(self, df, vocab, tag2idx, rel2idx):
                self.df = df
                self.vocab = vocab
                self.tag2idx = tag2idx
                self.rel2idx = rel2idx
                self.unkidx = vocab['<unk>']

            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                text = [self.vocab.get(token, self.unkidx) for token in row['utterances']]
                tags = [self.tag2idx[tag] for tag in row['IOB Slot tags']]
                # rels = [self.rel2idx[rel] for rel in row['Core Relations']]
                rels = row['Core Relations']
                return torch.Tensor(text).long(), torch.Tensor(tags).long(), torch.Tensor(rels).long()

            def __len__(self):
                return len(self.df)

        def collate(batch: list[tuple]):
            texts, tags, rels = zip(*batch)
            texts = pad_sequence(texts, batch_first=True, padding_value=vocab['<pad>'])
            tags = pad_sequence(tags, batch_first=True, padding_value=-100)
            rels = torch.stack(rels, dim=0)
            return texts, tags, rels

        ds = JointTrainingDataset(df, vocab, tag2idx, rel2idx)
        train_df, val_df = train_test_split(ds, test_size=0.01,random_state=42)
        dataloader_train = DataLoader(train_df, batch_size=parameters['batch_size'], collate_fn=collate)
        dataloader_val = DataLoader(val_df, batch_size=parameters['batch_size'], collate_fn=collate)

        class TransformerForIOBandRelation(nn.Module):
            def __init__(self, vocab_size, d_model, n_heads, num_encoder_layers, dim_feedforward, num_classes_iob, num_classes_relation, max_len, pad_token_id):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
                # self.pos_encoder = PositionalEncoding(d_model, max_len)
                self.transformer_encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward), 
                    num_layers=num_encoder_layers
                )
                self.iob_classifier = nn.Linear(d_model, num_classes_iob)
                self.relation_classifier = nn.Linear(d_model, num_classes_relation)
                self.d_model=d_model

            def forward(self, src, src_key_padding_mask):
                src = self.embedding(src) * math.sqrt(self.d_model)
                # src = self.pos_encoder(src)
                output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
                iob_output = self.iob_classifier(output)
                relation_output = self.relation_classifier(output.mean(dim=1))
                return iob_output, relation_output

        def create_src_key_padding_mask(src, pad_token_id):
            return src == pad_token_id

        def train(model, iterator, optimizer, tag_criterion, relation_criterion):
            model.train()
            epoch_loss, epoch_tag_acc, epoch_relation_acc = 0, 0, 0
            epoch_tag_loss, epoch_rel_loss = 0,0
            pad_token_id = 0

            for batch in tqdm(iterator):
                src, tags, relations = batch  # Adapt as per your data loader
                src_key_padding_mask = create_src_key_padding_mask(src, pad_token_id).to(device)
                optimizer.zero_grad()
                tag_output, relation_output = model(src.to(device), src_key_padding_mask.transpose(0,1).to(device))

                # Compute losses
                tag_loss = tag_criterion(tag_output.view(-1, tag_output.shape[-1]).to(device), tags.view(-1).to(device))
                relation_loss = relation_criterion(relation_output.to(device), relations.float().to(device))

                # Combine losses for backpropagation
                loss = (tag_loss*parameters['alpha'] + relation_loss*(1-parameters['alpha']))*2
                loss.backward()
                optimizer.step()

                # Compute accuracies
                tag_acc = calculate_accuracy_for_tags(tag_output.to(device), tags.to(device))
                relation_acc = calculate_relation_accuracy(relation_output.to(device), relations.to(device))

                epoch_loss += loss.item()
                epoch_tag_acc += tag_acc
                epoch_relation_acc += relation_acc
                epoch_tag_loss += tag_loss
                epoch_rel_loss += relation_loss

            return epoch_loss / len(iterator), epoch_tag_loss / len(iterator), epoch_rel_loss / len(iterator), epoch_tag_acc / len(iterator), epoch_relation_acc / len(iterator)

        def evaluate(model, iterator, tag_criterion, relation_criterion):
            model.eval()
            epoch_loss, epoch_tag_acc, epoch_relation_acc = 0, 0, 0
            epoch_tag_loss, epoch_rel_loss = 0,0
            batch_wise_true_labels_tag = []
            batch_wise_predictions_tag = []
            batch_wise_true_labels_rel = []
            batch_wise_predictions_rel = []
            pad_token_id=0

            with torch.no_grad():
                for batch in iterator:
                    src, tags, relations = batch
                    src_key_padding_mask = create_src_key_padding_mask(src, pad_token_id).to(device)
                    tag_output, relation_output = model(src.to(device), src_key_padding_mask.transpose(0,1).to(device))

                    tag_loss = tag_criterion(tag_output.view(-1, tag_output.shape[-1]).to(device), tags.view(-1).to(device))
                    relation_loss = relation_criterion(relation_output.to(device), relations.float().to(device))

                    loss = tag_loss + relation_loss

                    tag_acc = calculate_accuracy_for_tags(tag_output.to(device), tags.to(device))
                    relation_acc = calculate_relation_accuracy(relation_output.to(device), relations.to(device))

                    epoch_loss += loss.item()
                    epoch_tag_acc += tag_acc
                    epoch_relation_acc += relation_acc
                    epoch_tag_loss += tag_loss
                    epoch_rel_loss += relation_loss
                    batch_wise_true_labels_tag.append(tags.cpu())
                    batch_wise_predictions_tag.append(tag_output.cpu())
                    batch_wise_true_labels_rel.append(relations.cpu())
                    batch_wise_predictions_rel.append(relation_output.cpu())

            # flatten the list of predictions using itertools
            all_true_labels_tag = list(chain.from_iterable(batch_wise_true_labels_tag))
            all_predictions_tag = list(chain.from_iterable(batch_wise_predictions_tag))
            all_true_labels_rel = list(chain.from_iterable(batch_wise_true_labels_rel))
            all_predictions_rel = list(chain.from_iterable(batch_wise_predictions_rel))


            return epoch_loss / len(iterator), epoch_tag_loss / len(iterator), epoch_rel_loss / len(iterator), epoch_tag_acc / len(iterator), epoch_relation_acc / len(iterator), all_true_labels_tag,all_predictions_tag,all_true_labels_rel,all_predictions_rel

        # Define a function to calculate accuracy
        def calculate_accuracy_for_tags(predictions, true_labels):
            # Transfer the prediction data to real data
            _, predicted_labels = predictions.max(dim=2)
            # mask the paddings
            mask = true_labels != -100
            # calculate the correct predictions
            correct_predictions = (predicted_labels == true_labels) & mask
            # calculate accuracy
            accuracy = correct_predictions.sum().float() / mask.sum().float()
            return accuracy

        def calculate_relation_accuracy(predictions, true_labels):
            # Transfer predicted labels to binary labels
            predicted_labels = torch.sigmoid(predictions) > 0.5
            # calculate the correct predictions
            correct_predictions = (predicted_labels == true_labels.bool())
            # calculate accuracy
            accuracy = correct_predictions.sum().float() / correct_predictions.numel()
            return accuracy

        def print_classification_report_tag(predictions, true_labels):
            true_labels_text=[]
            pred_labels_text=[]
            for preds in predictions:
                pred_labels_text_sentence=[]
                predicted_labels=torch.argmax(F.softmax(preds,dim=-1),dim=-1).tolist()
                for index in predicted_labels:
                    pred_labels_text_sentence.append(idx2tag[index])
                pred_labels_text.append(pred_labels_text_sentence)
            for label in true_labels:
                true_labels_text_sentence=[]
                for index in label.tolist():
                    if index != -100:
                        true_labels_text_sentence.append(idx2tag[index])
                    else:
                        true_labels_text_sentence.append('pad')
                true_labels_text.append(true_labels_text_sentence)
            for i in range(len(pred_labels_text)):
                if len(pred_labels_text[i])!=len(true_labels_text[i]):
                    print('not consistant')
            print(classification_report_tag(true_labels_text, pred_labels_text,scheme=IOB2))

        def print_classification_report_rel(predictions, true_labels):
            true_labels_text=[]
            pred_labels_text=[]
            true_labels_index=[]
            for preds in predictions:
                pred_labels_text_sentence=[]
                predicted_labels = torch.sigmoid(preds) > 0.1
                for label in predicted_labels:
                    if label.item() == True:
                        pred_labels_text_sentence.append(int(1))
                    elif label.item() == False:
                        pred_labels_text_sentence.append(int(0))
                pred_labels_text.append(pred_labels_text_sentence)
            for label in true_labels:
                true_label_text_sentence=[]
                true_label_index_sentence=[]
                for i in range(len(label)):
                    if label[i]==1:
                        true_label_text_sentence.append(idx2rel[i])
                for i in label:
                    true_label_index_sentence.append(int(i))
                true_labels_index.append(true_label_index_sentence)
                true_labels_text.append(true_label_text_sentence)

            print(classification_report_rel(np.array(true_labels_index, dtype=np.float32), np.array(pred_labels_text, dtype=np.float32),target_names=labels_rel))

        # Define model parameters
        vocab_size = len(idx2token)  # Adjust as per your vocabulary
        d_model = parameters['d_model']
        n_heads = parameters['n_heads']
        num_encoder_layers = parameters['num_encoder_layers']
        dim_feedforward = parameters['hidden_dim']
        num_classes_iob = len(idx2tag)  # For IOB tagging (e.g., B, I, O)
        num_classes_relation = len(idx2rel)  # Adjust based on your 'Core Relations' classes
        max_len = parameters['max_len']  # Adjust as per your maximum sequence length
        pad_token_id = parameters['pad_token_id']  # Adjust as per your pad token id

        # Create the model
        model = TransformerForIOBandRelation(vocab_size, d_model, n_heads, num_encoder_layers, dim_feedforward, num_classes_iob, num_classes_relation, max_len, pad_token_id).to(device)

        # Example input (adjust the shape as per your data)
        # sequence length = the length of the sentence i'm passing in
        optimizer = optim.Adam(model.parameters(), lr=parameters['learning_rate'])
        criterion_tag = nn.CrossEntropyLoss(ignore_index=-100)
        criterion_relation = nn.BCEWithLogitsLoss()

        for epoch in range(parameters['epochs']):
            train_loss, train_tag_loss, train_relation_loss, train_tag_acc, train_relation_acc = train(model, dataloader_train, optimizer, criterion_tag, criterion_relation)

            print(f'Epoch: {epoch+1:02}')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Tag Loss: {train_tag_loss:.3f} | Train Relation Loss: {train_relation_loss:.3f}| Train Tag Acc: {train_tag_acc*100:.2f}% | Train Relation Acc: {train_relation_acc*100:.2f}%')

        valid_loss, val_tag_loss, val_relation_loss, valid_tag_acc, valid_relation_acc,all_true_labels_tag,all_predictions_tag,all_true_labels_rel, all_predictions_rel = evaluate(model, dataloader_val, criterion_tag, criterion_relation)
        print(f'\t Val Loss: {valid_loss:.3f} | Val Tag Loss: {val_tag_loss:.3f} | Val Relation Loss: {val_relation_loss:.3f}|  Val Tag Acc: {valid_tag_acc*100:.2f}% |  Val Relation Acc: {valid_relation_acc*100:.2f}%')
        print_classification_report_tag(all_predictions_tag,all_true_labels_tag)
        print_classification_report_rel(all_predictions_rel,all_true_labels_rel)

        torch.save(model.state_dict(), args.save_model)
        my_data = {'idx2tag': idx2tag, 'idx2rel': idx2rel, 'idx2token': idx2token,
                   'vocab_size':vocab_size, 'num_classes_iob':num_classes_iob, 
                   'num_classes_relation':num_classes_relation,'vocab':vocab,
    }
        with open('./my_saved_data.pkl', 'wb') as file:
            pickle.dump(my_data, file)


    if args.test:
        # test_model(args.data, args.model_path, args.output)
        device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
        def tokenize(s):
            return s.split()
        class TransformerForIOBandRelation(nn.Module):
            def __init__(self, vocab_size, d_model, n_heads, num_encoder_layers, dim_feedforward, num_classes_iob, num_classes_relation, max_len, pad_token_id):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
                # self.pos_encoder = PositionalEncoding(d_model, max_len)
                self.transformer_encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward), 
                    num_layers=num_encoder_layers
                )
                self.iob_classifier = nn.Linear(d_model, num_classes_iob)
                self.relation_classifier = nn.Linear(d_model, num_classes_relation)
                self.d_model=d_model

            def forward(self, src, src_key_padding_mask):
                src = self.embedding(src) * math.sqrt(self.d_model)
                # src = self.pos_encoder(src)
                output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
                iob_output = self.iob_classifier(output)
                relation_output = self.relation_classifier(output.mean(dim=1))
                return iob_output, relation_output
        with open('my_saved_data.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
        idx2tag=loaded_data['idx2tag']
        idx2rel=loaded_data['idx2rel']
        
        test_data = pd.read_csv(args.data)
        test_data['utterances'] = test_data['utterances'].apply(tokenize)
        vocab = loaded_data['vocab']
        tag2idx = {v:k for k,v in idx2tag.items()}
        rel2idx = {v:k for k,v in idx2rel.items()}

        vocab_size = loaded_data['vocab_size'] # Adjust as per your vocabulary
        d_model = parameters['d_model']
        n_heads = parameters['n_heads']
        num_encoder_layers = parameters['num_encoder_layers']
        dim_feedforward = parameters['hidden_dim']
        num_classes_iob = loaded_data['num_classes_iob']  # For IOB tagging (e.g., B, I, O)
        num_classes_relation = loaded_data['num_classes_relation']  # Adjust based on your 'Core Relations' classes
        max_len = parameters['max_len']  # Adjust as per your maximum sequence length
        pad_token_id = parameters['pad_token_id'] 

        model = TransformerForIOBandRelation(vocab_size, d_model, n_heads, num_encoder_layers, dim_feedforward, num_classes_iob, num_classes_relation, max_len, pad_token_id).to(device)
        model.load_state_dict(torch.load(args.model_path))

        class JointTrainingDataset_test(Dataset):
            def __init__(self, df, vocab, tag2idx, rel2idx):
                self.df = df
                self.vocab = vocab
                self.tag2idx = tag2idx
                self.rel2idx = rel2idx
                self.unkidx = vocab['<unk>']

            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                text = [self.vocab.get(token, self.unkidx) for token in row['utterances']]
                return torch.Tensor(text).long()

            def __len__(self):
                return len(self.df)
        def collate_test(batch: list[tuple]):
            texts = batch
            texts = pad_sequence(texts, batch_first=True, padding_value=vocab['<pad>'])
            return texts
        
        def create_src_key_padding_mask(src, pad_token_id):
            return src == pad_token_id
        
        test_df = JointTrainingDataset_test(test_data, vocab, tag2idx, rel2idx)
        dataloader_test = DataLoader(test_df, batch_size=1, collate_fn=collate_test)



        test_output_tags=[]
        test_output_rels=[]
        model.eval()

        with torch.no_grad():
            for batch in dataloader_test:
                src = batch
                src_key_padding_mask = create_src_key_padding_mask(src, pad_token_id).to(device)
                tag_output, relation_output = model(src.to(device), src_key_padding_mask.transpose(0,1).to(device))
                test_output_tags.append(tag_output.cpu())
                test_output_rels.append(relation_output.cpu())
        test_output_tags_flatten=list(chain.from_iterable(test_output_tags))
        test_output_rels_flatten=list(chain.from_iterable(test_output_rels))
        # Transfer predicted tags to real text tags
        pred_labels_text_tag=[]
        for preds in test_output_tags_flatten:
            pred_labels_text_sentence=[]
            predicted_labels=torch.argmax(F.softmax(preds,dim=-1),dim=-1).tolist()
            for index in predicted_labels:
                pred_labels_text_sentence.append(idx2tag[index])
            pred_labels_text_tag.append(pred_labels_text_sentence)
        # Transfer predicted relations to real text relations
        pred_labels_text_rel=[]
        for preds in test_output_rels_flatten:
            pred_labels_text_sentence=[]
            predicted_labels = torch.sigmoid(preds) > 0.1
            for i in range(len(predicted_labels)-1):
                if predicted_labels[i].item()==True:
                    pred_labels_text_sentence.append(idx2rel[i])
            pred_labels_text_rel.append(pred_labels_text_sentence)
        for i in pred_labels_text_rel:
            if len(i)!=1 and 'none' in i:
                i.remove('none')
        # Load the existing CSV file
        file_path = args.data
        new_path = args.output
        df = pd.read_csv(file_path)
        # Add these lists as new columns to the DataFrame
        df['IOB Slot tags'] = [' '.join(map(str, sublist)) if isinstance(sublist, list) else str(sublist) for sublist in pred_labels_text_tag]
        df['Core Relations'] = [' '.join(map(str, sublist)) if isinstance(sublist, list) else str(sublist) for sublist in pred_labels_text_rel]
        # Write the modified DataFrame back to the CSV file
        df.to_csv(new_path, index=False)



if __name__ == "__main__":
    main()
