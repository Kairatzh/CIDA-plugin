"""
hf.py — Интеграция CIDA-Plugin с Hugging Face Models.

Позволяет обернуть любую модель классификации последовательностей (Sequence Classification) 
в CIDA-Plugin в одну строчку кода, заменяя стандартную линейную голову.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any

try:
    from transformers.modeling_outputs import SequenceClassifierOutput
    _HAS_TRANSFORMERS = True
except ImportError:
    # Заглушка на случай отсутствия transformers (например, при тестировании голого PyTorch)
    class SequenceClassifierOutput:
        def __init__(self, loss=None, logits=None, hidden_states=None, attentions=None):
            self.loss = loss
            self.logits = logits
            self.hidden_states = hidden_states
            self.attentions = attentions
    _HAS_TRANSFORMERS = False

from .core import CIDAPlugin
from .config import CIDAPluginConfig
from .losses import CIDALoss


class HFModelWithCIDA(nn.Module):
    """
    Обертка над предобученной Hugging Face моделью классификации текстов.
    Перехватывает скрытые состояния базового энкодера и пропускает их через CIDA-Plugin.
    """
    def __init__(
        self,
        hf_model: nn.Module,
        cida_config: Optional[CIDAPluginConfig] = None,
        **cida_kwargs: Any
    ):
        super().__init__()
        
        # Сохраняем исходную конфигурацию
        self.config = getattr(hf_model, "config", None)
        
        # 1. Извлекаем базовый энкодер (transformer backbone)
        if hasattr(hf_model, "base_model"):
            self.encoder = hf_model.base_model
        else:
            # Fallback для кастомных моделей
            self.encoder = hf_model
            
        # 2. Определяем размерности
        if self.config is not None:
            d_model = getattr(self.config, "hidden_size", None) or getattr(self.config, "d_model", None)
            num_labels = getattr(self.config, "num_labels", 2)
        else:
            d_model = cida_kwargs.get("d_input")
            num_labels = cida_kwargs.get("num_classes", 2)
            
        if d_model is None:
            raise ValueError(
                "Не удалось автоматически определить hidden_size из конфигурации модели HF. "
                "Пожалуйста, передайте параметр d_input явно в cida_kwargs."
            )
            
        # 3. Инициализируем CIDA-Plugin
        if cida_config is None:
            cida_config = CIDAPluginConfig(
                d_input=d_model,
                num_classes=num_labels,
                **cida_kwargs
            )
        else:
            # Обновляем поля, если они переданы пустые
            if cida_config.d_input is None:
                cida_config.d_input = d_model
            if cida_config.num_classes is None:
                cida_config.num_classes = num_labels
                
        self.plugin = CIDAPlugin(cida_config)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any
    ) -> Union[SequenceClassifierOutput, tuple]:
        
        use_return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", True)
        
        # 1. Прогоняем через базовый энкодер
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=use_return_dict,
            **kwargs
        )
        
        # 2. Извлекаем sequence outputs
        if isinstance(encoder_outputs, tuple):
            last_hidden_state = encoder_outputs[0]
        else:
            last_hidden_state = encoder_outputs.last_hidden_state
            
        # Извлекаем CLS токен (первый токен последовательности) как pooled representation
        pooled_output = last_hidden_state[:, 0, :]
        
        # 3. Вызываем CIDA-Plugin
        plugin_outputs = self.plugin(
            pooled=pooled_output,
            seq_output=last_hidden_state,
            mask=attention_mask
        )
        
        logits = plugin_outputs["p_final"]
        
        # 4. Вычисляем лосс при наличии меток
        loss = None
        if labels is not None:
            loss_fn = CIDALoss(
                multi_label=self.plugin.config.multi_label,
                lambda_cal=self.plugin.config.lambda_cal,
                lambda_ac=self.plugin.config.lambda_ac,
                min_disagreement=self.plugin.config.min_disagreement
            ).to(logits.device)
            
            orth_loss = plugin_outputs.get("L_orth", None)
            
            loss, _ = loss_fn(
                p_final=logits,
                y=labels,
                b_all=plugin_outputs["b_all"],
                orth_loss=orth_loss
            )
            
        # 5. Возвращаем результат
        if not use_return_dict:
            output = (logits,)
            if isinstance(encoder_outputs, tuple):
                output = output + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output
            
        # Собираем SequenceClassifierOutput для HF-совместимости
        res = SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=getattr(encoder_outputs, "hidden_states", None),
            attentions=getattr(encoder_outputs, "attentions", None)
        )
        
        # Добавляем CIDA метаданные как атрибуты для диагностики
        res.b_all = plugin_outputs["b_all"]
        res.uncertainty = plugin_outputs["uncertainty"]
        res.disagreement = plugin_outputs["disagreement"]
        res.rounds_used = plugin_outputs["rounds_used"]
        res.plugin_outputs = plugin_outputs
        
        return res


def wrap_hf_model(
    hf_model: nn.Module,
    cida_config: Optional[CIDAPluginConfig] = None,
    **kwargs: Any
) -> HFModelWithCIDA:
    """
    Быстрая обертка для Hugging Face sequence classification моделей.
    
    Пример использования:
        from transformers import AutoModelForSequenceClassification
        from cida_plugin import wrap_hf_model

        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        model = wrap_hf_model(model, max_rounds=3)
        
        # Теперь модель готова к обучению с CIDA-Plugin в Hugging Face Trainer!
    """
    return HFModelWithCIDA(hf_model, cida_config=cida_config, **kwargs)


class HFModelForCausalLMWithCIDA(nn.Module):
    """
    Wrapper for Autoregressive/Causal Language Models (like Qwen, Gemma, Llama).
    Intercepts the prompt input, runs CIDA deliberation on the latent space,
    projects the final agent consensus into plan token embeddings,
    prepends them to the input embeddings, and uses the base model to generate.
    """
    def __init__(self, hf_model: nn.Module, cida_plugin: nn.Module, plan_projector: nn.Module):
        super().__init__()
        self.base_model = hf_model
        self.plugin = cida_plugin
        self.plan_projector = plan_projector

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any
    ):
        if inputs_embeds is None and input_ids is not None:
            # 1. Run prompt embedding
            embed_tokens = self.base_model.get_input_embeddings()
            inputs_embeds = embed_tokens(input_ids)

            # 2. Extract CIDA consensus plan
            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False
                )
                last_hidden = outputs.hidden_states[-1]
                sidx = attention_mask.sum(dim=1) - 1
                bidx = torch.arange(input_ids.size(0), device=input_ids.device)
                pooled_hidden = last_hidden[bidx, sidx]

            cida_out = self.plugin(pooled=pooled_hidden)
            plan_embeddings = self.plan_projector(cida_out['state_final'].s) # (B, P, H)

            # Prepend plan embeddings to original tokens
            inputs_embeds = torch.cat([plan_embeddings, inputs_embeds], dim=1)
            
            # Prepend ones to attention mask for plan tokens
            B, P, _ = plan_embeddings.shape
            plan_mask = torch.ones(B, P, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([plan_mask, attention_mask], dim=1)
            
            # Clear input_ids as we use inputs_embeds
            input_ids = None

        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs
        )

    def generate_with_cida(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        **generation_kwargs: Any
    ) -> torch.Tensor:
        """
        Runs CIDA multi-agent deliberation on the prompt and prepends
        the consensus plan embeddings before generating the final text response.
        """
        embed_tokens = self.base_model.get_input_embeddings()
        prompt_embeds = embed_tokens(input_ids) # (B, Lq, H)

        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False
            )
            last_hidden = outputs.hidden_states[-1]
            sidx = attention_mask.sum(dim=1) - 1
            bidx = torch.arange(input_ids.size(0), device=input_ids.device)
            pooled_hidden = last_hidden[bidx, sidx]

        cida_out = self.plugin(pooled=pooled_hidden)
        plan_embeddings = self.plan_projector(cida_out['state_final'].s) # (B, P, H)

        # Concatenate plan and prompt embeddings
        inputs_embeds = torch.cat([plan_embeddings, prompt_embeds], dim=1)

        # Prepend plan mask to attention mask
        B, P, _ = plan_embeddings.shape
        plan_mask = torch.ones(B, P, dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([plan_mask, attention_mask], dim=1)

        # Generate tokens
        generated_ids = self.base_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **generation_kwargs
        )
        return generated_ids


def wrap_hf_causal_lm(
    hf_model: nn.Module,
    cida_plugin: nn.Module,
    plan_projector: nn.Module
) -> HFModelForCausalLMWithCIDA:
    """
    Wraps a Causal LM (Qwen/Llama) with CIDA and a PlanProjector for generative inference.
    """
    return HFModelForCausalLMWithCIDA(hf_model, cida_plugin, plan_projector)

