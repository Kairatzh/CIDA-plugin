import os
import torch
from huggingface_hub import hf_hub_download

class LatentPrompt:
    """
    Управляет сохранением и загрузкой "Латентных Промптов" (векторных масок).
    Поддерживает Мульти-Векторную Память (разбиение гигантских промптов на чанки)
    для идеальной работы с механизмом Cross-Attention.
    """
    def __init__(self, tensors: list, metadata: dict = None):
        # Если передали один тензор, оборачиваем в список для обратной совместимости
        if isinstance(tensors, torch.Tensor):
            self.tensors = [tensors]
        else:
            self.tensors = tensors
        self.metadata = metadata or {}

    @property
    def tensor(self):
        """Обратная совместимость: возвращает первый (или единственный) тензор"""
        return self.tensors[0] if self.tensors else None

    def save_local(self, file_path: str):
        """Сохраняет мульти-векторный латентный промпт на диск."""
        torch.save({
            "tensors": [t.cpu() for t in self.tensors],
            "metadata": self.metadata
        }, file_path)
        print(f"Латентный промпт (Концепций: {len(self.tensors)}) успешно сохранен в {file_path}")

    @classmethod
    def load_local(cls, file_path: str, device="cuda"):
        """Загружает мульти-векторный латентный промпт с диска."""
        data = torch.load(file_path, map_location=device)
        
        # Обработка легаси-формата (когда сохраняли только 1 тензор под ключом "tensor")
        if "tensor" in data and "tensors" not in data:
            tensors = [data["tensor"].to(device)]
        else:
            tensors = [t.to(device) for t in data["tensors"]]
            
        return cls(tensors=tensors, metadata=data.get("metadata", {}))

    @classmethod
    def from_huggingface(cls, repo_id: str, filename: str, device="cuda"):
        """
        Загружает латентный промпт напрямую из хранилища на Hugging Face.
        """
        print(f"Загрузка латентного промпта из {repo_id}/{filename}...")
        file_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return cls.load_local(file_path, device)

    @classmethod
    def compile_from_text(cls, orchestrator, long_text_prompt: str, chunk_size: int = 500):
        """
        [Мульти-Векторная Компиляция]
        Превращает длинный текстовый промпт в набор сжатых латентных векторов (Latent Concepts).
        Текст разбивается на части (чанки). Модель переваривает каждую часть,
        извлекая суть, что гарантирует идеальную работу Cross-Attention при поиске правил.
        """
        tensors = []
        total_tokens = 0
        
        # Простая эвристика разбиения: по двойному переносу строки или по длине
        chunks = [c.strip() for c in long_text_prompt.split("\n\n") if c.strip()]
        if not chunks:
            chunks = [long_text_prompt]
            
        print(f"Компиляция Латентного Промпта. Всего концепций (чанков): {len(chunks)}")
        
        with torch.no_grad():
            for i, chunk in enumerate(chunks):
                enc = orchestrator.tokenizer(chunk, return_tensors="pt").to(orchestrator.device)
                total_tokens += len(enc.input_ids[0])
                
                out = orchestrator.llm(
                    input_ids=enc.input_ids, 
                    attention_mask=enc.attention_mask,
                    output_hidden_states=True
                )
                
                # Извлекаем скрытое состояние (суть конкретного правила/примера)
                latent_vector = out.hidden_states[-1][0, -1].float().unsqueeze(0)
                tensors.append(latent_vector)
                
        metadata = {
            "original_length": total_tokens,
            "num_concepts": len(tensors),
            "chunking_strategy": "double_newline"
        }
        
        print(f"Успешно скомпилировано! Токенов: {total_tokens} -> Векторов: {len(tensors)}")
        return cls(tensors=tensors, metadata=metadata)
