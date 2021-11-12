import torch
import torch.nn as nn
from transformers import BertModel


class BertFC(nn.Module):
    def __init__(self):
        """Downloads a BERT base uncased model and adds a linear layer on
        top of it"""
        super(BertFC, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # TODO(rajiv/Greg): Maybe add an intermediate layer before this.
        # Going from 768 -> 1 is not very ideal I guess.
        self.classifier = nn.Linear(768, 1)

    def forward(self, ids, token_ids, mask):
        """
        inputs: ids, token_ids, and mask each of dim = [bsz x seqlen]
        returns: probability of a positive label, dim = [bsz]
        """
        sequence_output = \
        self.bert(input_ids=ids, token_type_ids=token_ids, attention_mask=mask)[
            0]
        pooled_output = \
        self.bert(input_ids=ids, token_type_ids=token_ids, attention_mask=mask)[
            1]
        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        # sequence_output = nn.ReLU()(sequence_output)
        # sequence_output = torch.tanh(sequence_output)
        # linear_output = self.classifier(sequence_output[:, 0, :])
        output = self.classifier(pooled_output)
        return output.squeeze(0)

def modify_authors_state_dict(state_dict):
    """The state dicts prefixes don't match (ours is bert.xyz,
    their's is model.model.xyz. This function alters the
    naming in their state_dict to match"""
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for x in state_dict.items():
        # print(x)
        name = x[0]
        vals = x[1]
        if name[:11] == "model.model":
            new_name = "bert" + name[11:]
        else:
            new_name = name[6:]
        new_state_dict[new_name] = vals
    return new_state_dict


# Build the model
if __name__ == '__main__':
    bert_fc_model = BertFC()
    modified_state_dict = modify_authors_state_dict(authors_model['state_dict'])
    bert_fc_model.load_state_dict(authors_model['state_dict'])
    bert_fc_model.load_state_dict(modified_state_dict)
    authors_model['state_dict']
    authors_model = torch.load('data/FC_model.ckpt')

