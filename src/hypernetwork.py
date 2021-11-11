import torch.nn as nn
import torch.nn.functional as F

class hypernetwork(nn.Module):                                                                       
    def __init__(self,                                                                               
                 n,                                                                                  
                 m,                                                                                  
                 input_size=768,  # verified                          
                 hidden_size=128,  # verified                         
                 linear_out=1024,  # verified                         
                 num_layers=1):  # verified                           
        """                                                                                          
        Args:                                                                                        
            n:                                                                                       
            m:                                                                                       
            input_size:                                                                              
            hidden_size:                                                                             
            linear_out:                                                                              
            num_layers:                                                                              
        """                                                                                          
        super(hypernetwork, self).__init__()                          
                                                                                                     
        self.bilstm = nn.LSTM(input_size=input_size,                  
                              hidden_size=hidden_size,                
                              num_layers=num_layers,                  
                              bidirectional=True,
                              batch_first = True)     
        self.num_layers = num_layers
        self.hidden_size = hidden_size           
        self.linear = nn.Linear(2*hidden_size, linear_out)            
        self.alpha_linear = nn.Linear(linear_out, m)                 
        self.beta_linear = nn.Linear(linear_out, m)                  
        self.gamma_linear = nn.Linear(linear_out, n)                 
        self.delta_linear = nn.Linear(linear_out, n)                 
                                                                                                     
        # TODO(rajiv): Maybe an intermediate layer?                   
        self.eta_linear = nn.Linear(linear_out, 1)                   
                                                                                                     
    def forward(self, X, gradW):                                                                     
        """                                                                                          
        Args:                                                                                        
            X: Output from BERT 
            gradW: gradients of a weight matrix 
                in `finetuned_bert`, dim: N x M
        Returns:                                                                                     
            delW: deltas to BERT weights for the input matrix
                dim: N x M
        """                                                                                          

        # TODO(rajiv): Make sure X is input + y +  a and not just input.                                                                                                     
        # setting hidden states to allow lstm to run
        hidden = torch.zeros((2*self.num_layers, 1, self.hidden_size))
        cell   = torch.zeros((2*self.num_layers, 1, self.hidden_size))
        # TODO: With L of 512, is the BiLSTM model going to forget parameters 
        # through time? Can we mask the bilstm input?
        _, (hidden, _) = self.bilstm(X, (hidden, cell)) #hidden dim [2 x 1 x 128]
        output = torch.tanh(self.linear(hidden.flatten())) #[1024]

        alpha = self.alpha_linear(output) #[m]                          
        beta = self.beta_linear(output)   #[m]                         
        gamma = self.gamma_linear(output) #[n]                         
        delta = self.delta_linear(output) #[n]                            
        eta = self.eta_linear(output)     #[1]
        #print("gradW dim: ", gradW.size())
        #print("alpha dim: ", alpha.size())                               
        #print("beta dim: ", beta.size())
        #print("gamma dim: ", gamma.size())
        #print("delta dim: ", delta.size())
        #print("eta dim: ", eta.size())
                                                                                                     
        # TODO(rajiv): While computing *_hat, we are assuming that the first
        # dimension would be the batch dimension. So we transpose the last two
        # dimensions.                                                                                
        #Greg: I think this is done?

        alpha_hat = torch.outer(gamma, F.softmax(alpha))
        beta_hat = torch.outer(delta, F.softmax(beta))
        #print("alpha_hat size: ", alpha_hat.size())
        #print("beta_hat size: ", beta_hat.size())
                                                                                                     
        delW = torch.sigmoid(eta) * ((alpha_hat * gradW) + beta_hat)
        return delW
