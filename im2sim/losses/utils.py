# ==============================================================================
# Copyright 2026 University College London.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import inspect

from torch import nn


class GraphLoss(nn.Module):
    def __init__(self, loss_fn, kwargs):
        self.loss_fn = loss_fn
        self.params = inspect.signature(loss_fn).parameters
        self.kwargs = kwargs

    def forward(self, true_graph, pred_graph):
        gr_dict = {"true": true_graph, "pred": pred_graph}
        call_args = {
            key: getattr(
                gr_dict[key.split("_")[0]], key.split[1]
            )  # key is in format <true/pred>_<attr_name>
            for key in self.params
        }
        loss = self.loss_fn(**call_args, **self.kwargs)
        return loss
