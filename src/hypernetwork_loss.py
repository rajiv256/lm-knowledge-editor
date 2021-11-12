import torch


def kl_loss_test(target_probs, output_probs):
    """Computes KL(output_probs || target_probs)
    """
    loss_fn = torch.nn.KLDivLoss(reduction='sum', log_target=True)
    loss = loss_fn(output_probs.log(), target_probs.log())
    return loss


def kl_loss(O_x, bert, bert_edited):
    original_model_probs = bert(O_x)  # Gives len(O_x)*2 of probabilities.
    edited_model_probs = bert_edited(O_x)  # len(O_x)*2

    # TODO(rajiv/Greg): Recheck if these args are alright.
    # Is `reduction='sum'` alright?
    loss_fn = torch.nn.KLDivLoss(reduction='sum',
        log_target=True)

    # This loss function should have inputs the log-probabilities.
    # Here, original model is considered as `target` and we are measuring
    # kl-div of the `edited_model_probs` with respect to these targets.
    # For reference: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    # Look at this line: l(x, y) = y_n * (log(y_n) - x_n)
    # We can give the input in log probabilities in both x and y,
    # so log_target is set to "True" above.
    loss = loss_fn(edited_model_probs.log(), original_model_probs.log())
    return loss


if __name__=="__main__":
    output = torch.tensor([0.99, 0.01])
    target = torch.tensor([0.5, 0.5])
    loss = kl_loss_test(target, output)
    print(f'loss: {loss.item()}')
