"""BT-BERT implicit model implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from transformers import AutoModel, AutoConfig, PreTrainedModel


@dataclass
class BTBertConfig:
    pretrained_model_name: str = "bert-base-uncased"
    attention_scale: float = 16.0


class BTBertModel(nn.Module):
    """Implicit BT-BERT model (logits derived from attention weights)."""

    def __init__(self, config: BTBertConfig, loss_pos_weight: Optional[float] = None) -> None:
        super().__init__()
        hf_config = AutoConfig.from_pretrained(
            config.pretrained_model_name, output_attentions=True
        )
        self.encoder: PreTrainedModel = AutoModel.from_pretrained(
            config.pretrained_model_name, config=hf_config
        )
        self.attention_scale = config.attention_scale
        self.loss_pos_weight = loss_pos_weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
            return_dict=True,
        )

        attentions = outputs.attentions[-1]  # last layer -> (batch, heads, seq, seq)
        logits = self.attention_scale * attentions[:, :, 0, 0].sum(dim=1)

        result: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            loss_kwargs = {}
            if self.loss_pos_weight is not None:
                pos_weight = torch.tensor(
                    self.loss_pos_weight, dtype=logits.dtype, device=logits.device
                )
                loss_kwargs["pos_weight"] = pos_weight
            if sample_weight is not None:
                sample_weight = sample_weight.to(logits.device)
                loss_kwargs["weight"] = sample_weight
            loss_fn = nn.BCEWithLogitsLoss(**loss_kwargs)
            loss = loss_fn(logits, labels.float())
            result["loss"] = loss
        return result
