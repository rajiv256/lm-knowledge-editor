class BertFC(nn.Module):
    def __init__(self):
        """Downloads a BERT base uncased model and adds a linear layer on top of it"""
        super(BertFC, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(768, 1)

    def forward(self, ids, token_ids, mask):
        """
        inputs: ids, token_ids, and mask each of dim = [bsz x seqlen]
        returns: probability of a positive label, dim = [bsz]
        """
        sequence_output = self.bert(input_ids = ids, token_type_ids = token_ids, attention_mask = mask)[0]
        pooled_output = self.bert(input_ids = ids, token_type_ids = token_ids, attention_mask = mask)[1]
        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        # sequence_output = nn.ReLU()(sequence_output)
        # sequence_output = torch.tanh(sequence_output)
        # linear_output = self.classifier(sequence_output[:, 0, :])
        output = self.classifier(pooled_output)
        return output.squeeze(0)

# Build the model
if __name__ == '__main__':
    bert_fc_model = BertFC()

