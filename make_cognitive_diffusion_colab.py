import json
import os

def create_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🧠 Kairos Cognitive Diffusion: CIDA + DiffusionGemma\n",
                    "\n",
                    "Добро пожаловать в практический ноутбук по интеграции плагина **CIDA (Cognitive Interactive Deliberation Agents)** с новейшей дискретной диффузионной языковой моделью от Google DeepMind — **DiffusionGemma**.\n",
                    "\n",
                    "### Цель этого ноутбука:\n",
                    "1. Загрузить официальную предобученную модель `google/diffusiongemma-26B-A4B-it` с Hugging Face.\n",
                    "2. Применить 4-битное квантование (`bitsandbytes`), чтобы модель поместилась в VRAM стандартной Colab GPU (L4 или T4).\n",
                    "3. Настроить когнитивную обертку `CognitiveDiffusionGemma`, которая внедряет плагин CIDA прямо в процесс генерации (denoising).\n",
                    "4. Продемонстрировать работу многоагентных дебатов для повышения связности параллельной генерации текста.\n",
                    "\n",
                    "---\n",
                    "### ⚠️ ВАЖНОЕ ПРИМЕЧАНИЕ:\n",
                    "Модель `DiffusionGemma` является закрытой (gated). Перед запуском:\n",
                    "1. Войдите в свой аккаунт Hugging Face.\n",
                    "2. Перейдите на страницу [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) и подтвердите согласие с лицензионным соглашением.\n",
                    "3. Создайте HF Access Token в настройках аккаунта."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Подготовка окружения и клонирование репозитория"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Клонируем репозиторий с исходным кодом CIDA\n",
                    "!git clone https://github.com/Kairatzh/CIDA-plugin.git\n",
                    "%cd CIDA-plugin"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Устанавливаем зависимости\n",
                    "!pip install -q transformers accelerate bitsandbytes huggingface_hub"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Авторизация в Hugging Face"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Введите ваш токен от Hugging Face для загрузки весов\n",
                    "from huggingface_hub import notebook_login\n",
                    "notebook_login()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Загрузка DiffusionGemma (4-bit)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import torch\n",
                    "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n",
                    "\n",
                    "model_id = \"google/diffusiongemma-26B-A4B-it\"\n",
                    "\n",
                    "# Настраиваем 4-битное квантование для экономии VRAM\n",
                    "quantization_config = BitsAndBytesConfig(\n",
                    "    load_in_4bit=True,\n",
                    "    bnb_4bit_compute_dtype=torch.float16,\n",
                    "    bnb_4bit_quant_type=\"nf4\"\n",
                    ")\n",
                    "\n",
                    "print(\"Загружаем токенизатор...\")\n",
                    "tokenizer = AutoTokenizer.from_pretrained(model_id)\n",
                    "\n",
                    "print(\"Загружаем модель DiffusionGemma (это может занять несколько минут)...\")\n",
                    "base_model = AutoModelForCausalLM.from_pretrained(\n",
                    "    model_id,\n",
                    "    quantization_config=quantization_config,\n",
                    "    device_map=\"auto\"\n",
                    ")\n",
                    "print(\"Модель успешно загружена!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Обертка CognitiveDiffusionGemma\n",
                    "\n",
                    "Мы создаем класс, который перехватывает скрытые состояния последнего слоя Gemma MoE, передает их агентам CIDA для дебатов по всей последовательности, а затем проецирует консенсус обратно в оригинальный `lm_head` модели Gemma."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import torch.nn as nn\n",
                    "from cida_plugin.core import CIDAPlugin\n",
                    "from cida_plugin.config import CIDAPluginConfig\n",
                    "\n",
                    "class CognitiveDiffusionGemma(nn.Module):\n",
                    "    def __init__(self, base_model, d_hidden=256):\n",
                    "        super().__init__()\n",
                    "        self.base_model = base_model\n",
                    "        self.d_model = base_model.config.hidden_size\n",
                    "        \n",
                    "        # Инициализируем CIDA\n",
                    "        # Мы используем dummy num_classes=2, так как классификацию токенов \n",
                    "        # делает оригинальный lm_head модели Gemma, а CIDA оптимизирует скрытые состояния\n",
                    "        self.cida_config = CIDAPluginConfig(\n",
                    "            d_input=self.d_model,\n",
                    "            d_hidden=d_hidden,\n",
                    "            num_classes=2,\n",
                    "            multi_label=False,\n",
                    "            max_rounds=2\n",
                    "        )\n",
                    "        self.cida = CIDAPlugin(self.cida_config)\n",
                    "        \n",
                    "        # Проекция из консенсуса CIDA обратно в пространство Gemma d_model\n",
                    "        self.out_proj = nn.Linear(d_hidden, self.d_model)\n",
                    "        \n",
                    "    def forward(self, input_ids, attention_mask=None):\n",
                    "        B, L = input_ids.shape\n",
                    "        \n",
                    "        # 1. Прогоняем данные через оригинальные MoE слои\n",
                    "        with torch.no_grad():\n",
                    "            outputs = self.base_model.model(\n",
                    "                input_ids=input_ids,\n",
                    "                attention_mask=attention_mask,\n",
                    "                output_hidden_states=True,\n",
                    "                return_dict=True\n",
                    "            )\n",
                    "        # Скрытые состояния последнего слоя\n",
                    "        H = outputs.hidden_states[-1] # (B, L, d_model)\n",
                    "        \n",
                    "        # 2. Готовим данные для CIDA\n",
                    "        # Чтобы каждый токен общался со всей последовательностью, \n",
                    "        # мы дублируем контекст для каждого токена\n",
                    "        H_seq = H.unsqueeze(1).expand(B, L, L, self.d_model).reshape(B * L, L, self.d_model)\n",
                    "        H_flat = H.reshape(B * L, self.d_model)\n",
                    "        \n",
                    "        # 3. Запускаем когнитивные дебаты\n",
                    "        cida_out = self.cida(pooled=H_flat, seq_output=H_seq)\n",
                    "        \n",
                    "        # Извлекаем финальные состояния агентов\n",
                    "        # state_final.s shape: (B * L, M, d_hidden)\n",
                    "        s_final = cida_out[\"state_final\"].s.mean(dim=1) # (B * L, d_hidden)\n",
                    "        \n",
                    "        # 4. Проецируем обратно в пространство Gemma и восстанавливаем размерность последовательности\n",
                    "        H_refined = self.out_proj(s_final).view(B, L, self.d_model)\n",
                    "        \n",
                    "        # 5. Классифицируем через оригинальный lm_head\n",
                    "        logits = self.base_model.lm_head(H_refined) # (B, L, vocab_size)\n",
                    "        \n",
                    "        return logits, cida_out[\"disagreement\"].view(B, L)\n",
                    "\n",
                    "print(\"Когнитивная обертка успешно определена!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Запуск Теста\n",
                    "\n",
                    "Мы создаем экземпляр когнитивной диффузии и делаем пробный шаг генерации на случайном промпте."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                    "cognitive_model = CognitiveDiffusionGemma(base_model).to(device)\n",
                    "\n",
                    "prompt = \"The key to artificial general intelligence is\"\n",
                    "inputs = tokenizer(prompt, return_tensors=\"pt\").to(device)\n",
                    "\n",
                    "# Выполняем когнитивный forward pass\n",
                    "logits, disagreement = cognitive_model(inputs[\"input_ids\"], inputs.get(\"attention_mask\"))\n",
                    "\n",
                    "print(f\"Входные токены: {inputs['input_ids'].shape}\")\n",
                    "print(f\"Выходные логиты: {logits.shape}\")\n",
                    "print(f\"Среднее несогласие агентов CIDA на токен: {disagreement.mean().item():.4f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Визуализация разнообразия агентов\n",
                    "\n",
                    "Давайте проверим, насколько мнения агентов расходятся в зависимости от токенов промпта. Высокое несогласие обычно сигнализирует о сложной семантической развилке."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "tokens = tokenizer.convert_ids_to_tokens(inputs[\"input_ids\"][0])\n",
                    "dis_values = disagreement[0].detach().cpu().numpy()\n",
                    "\n",
                    "print(\"Анализ неопределенности токенов:\")\n",
                    "print(\"-\" * 50)\n",
                    "for tok, val in zip(tokens, dis_values):\n",
                    "    print(f\"Токен: {tok:<15} | Несогласие (Disagreement): {val:.4f}\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open("Kairos_Cognitive_Diffusion.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print("Notebook 'Kairos_Cognitive_Diffusion.ipynb' successfully created!")

if __name__ == "__main__":
    create_notebook()
