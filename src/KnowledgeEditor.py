import torch
import torch.nn as nn

from hypernetwork import HyperNetwork


def get_attributes(module, names):
    """"
  inputs: Base Module and a list of module names
  returns: the corresponding module
  """
    if len(names)==1:
        return getattr(module, names[0])
    else:
        return get_attributes(getattr(module, names[0]), names[1:])


class KnowledgeEditor(nn.Module):
    def __init__(self, BERT_model):
        """
    given a bert model, set up a hypernetwork                     
    for every non-bias, embedding, or layer-norm                  
    """
        super(KnowledgeEditor, self).__init__()
        self.bert = BERT_model
        self.hyper_network_dict = nn.ModuleDict()
        for name, layer in BERT_model.named_parameters():
            if ("LayerNorm" not in name and
                    "bias" not in name and
                    "embed" not in name):
                # Layers of size NxM
                h = HyperNetwork(layer.size()[0], layer.size()[1])
                self.hyper_network_dict[str(name).replace(".", "-")] = h


    def forward(
            self, X_ids, X_type_ids, X_mask, A,
            X_Y_A_ids, X_Y_A_type_ids, X_Y_A_mask
    ):
        """takes tokenized input X, alternative answer A, and
    hypernetwork specialized input X-<SEP>-Y-<SEP>-A                        
    returns updated parameters in a dictionary of {Layer_name: delta_params}
    """
        # run the BERT model forward
        self.bert.zero_grad()
        outputs = self.bert(X_ids, X_type_ids, X_mask)
        bert_loss = torch.nn.BCEWithLogitsLoss()
        bert_losses = bert_loss(outputs, A)
        bert_losses.backward()  # get the gradients of BERT
        # get pooled encoding from BERT
        encoded_input = self.bert.bert(X_Y_A_ids, X_Y_A_type_ids, X_Y_A_mask)[0]

        # Run the hypernetwork
        h_outputs = {}
        for layer, network in self.hyper_network_dict.items():
            # get the grad for the layer in question
            gradW = get_attributes(self.bert, layer.split("-") + ['grad'])
            # puts the new parameters in a dictionary of layer names
            h_outputs[layer] = network(encoded_input, gradW)

        return h_outputs


if __name__=='__main__':
    ke = KnowledgeEditor(bert_fc_model)
