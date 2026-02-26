from torch import nn 
import inspect

class GraphLoss(nn.Module):

    def __init__(self, loss_fn, kwargs):
        self.loss_fn = loss_fn 
        self.params = inspect.signature(loss_fn).parameters
        self.kwargs = kwargs

    
    def forward(self,true_graph, pred_graph):
        gr_dict = {'true':true_graph, 'pred':pred_graph}
        call_args = {key: getattr(gr_dict[key.split('_')[0]], key.split[1]) # key is in format <true/pred>_<attr_name>
                     for key in self.params}
        loss = self.loss_fn(**call_args, **self.kwargs)
        return loss