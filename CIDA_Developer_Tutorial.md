# CIDA Inference Library
## Полное Руководство | Comprehensive Guide | 综合指南

**[RU]** Добро пожаловать в официальное руководство по библиотеке `cida_inference`. Этот документ проведет вас по всему жизненному циклу использования библиотеки: от установки до масштабирования Enterprise-пайплайнов.
**[EN]** Welcome to the official guide for the `cida_inference` library. This document will guide you through the entire lifecycle of using the library: from installation to scaling Enterprise pipelines.
**[ZH]** 欢迎使用 `cida_inference` 库官方指南。本文档将引导您完成使用该库的整个生命周期：从安装到扩展企业管道。

---

### 1. Как скачать и установить | How to Download and Install | 如何下载和安装

**[RU]** Библиотека CIDA устанавливается стандартными средствами Python. Поскольку она тесно интегрирована с PyTorch и HuggingFace, убедитесь, что в вашей среде установлены драйверы CUDA.
**[EN]** The CIDA library is installed using standard Python tools. Since it is tightly integrated with PyTorch and HuggingFace, ensure that CUDA drivers are installed in your environment.
**[ZH]** CIDA 库使用标准 Python 工具安装。由于它与 PyTorch 和 HuggingFace 紧密集成，请确保您的环境中已安装 CUDA 驱动程序。

```bash
# [RU] Установка ядра CIDA и инференс-компонентов из PyPI
# [EN] Installing CIDA core and inference components from PyPI
# [ZH] 从 PyPI 安装 CIDA 核心和推理组件
pip install cida-plugin

# [RU] Установка необходимых зависимостей (PyTorch, Transformers)
# [EN] Installing required dependencies (PyTorch, Transformers)
# [ZH] 安装所需的依赖项 (PyTorch, Transformers)
pip install torch transformers
```

---

### 2. Запуск и базовое использование | Running and Basic Usage | 运行和基本用法

**[RU]** Для запуска агента необходимо инициализировать `KairosOrchestrator` (который скачает базовую модель и адаптер CIDA) и передать ему ваши функции в качестве инструментов.
**[EN]** To run an agent, you need to initialize the `KairosOrchestrator` (which will download the base model and the CIDA adapter) and pass your functions to it as tools.
**[ZH]** 要运行代理，您需要初始化 `KairosOrchestrator`（它将下载基础模型和 CIDA 适配器）并将您的函数作为工具传递给它。

```python
import torch
from cida_plugin.orchestrator import KairosOrchestrator
from cida_plugin.inference.latent_agent import LatentAgent

# [RU] Загрузка ядра и адаптера
# [EN] Loading the core and adapter
# [ZH] 加载核心和适配器
orchestrator = KairosOrchestrator(
    backbone_model_name="Qwen/Qwen2.5-7B-Instruct",
    cida_model_path_or_repo="Kairatzh/CIDA-plugin-v1",
    device="cuda",
    torch_dtype=torch.bfloat16
)

# [RU] Создание простого инструмента
# [EN] Creating a simple tool
# [ZH] 创建一个简单的工具
def get_system_status() -> str:
    """Returns the current system status."""
    return "System is operational. CPU: 45%, RAM: 60%."

# [RU] Инициализация агента и запуск
# [EN] Initializing the agent and running
# [ZH] 初始化代理并运行
agent = LatentAgent(
    wrapper=orchestrator,
    llm=orchestrator.llm,
    tokenizer=orchestrator.tokenizer,
    tools={"get_system_status": get_system_status}
)

print(agent.run("Check the server parameters."))
```

---

### 3. Преимущества архитектуры CIDA | Advantages of the CIDA Architecture | CIDA 架构的优势

**[RU]** В отличие от стандартных LLM-агентов, CIDA работает на уровне скрытых состояний (System 2).
1. **Точность:** Векторные дебаты перед генерацией текста исключают галлюцинации в синтаксисе тулов.
2. **Экономия:** Агент думает внутри матрицы, не генерируя тысячи текстовых токенов `Thought:`, что кардинально снижает стоимость вывода.
3. **Безопасность:** Латентный фаервол (Latent Firewall) блокирует вредоносные намерения еще до того, как модель начнет их озвучивать.

**[EN]** Unlike standard LLM agents, CIDA operates at the level of hidden states (System 2).
1. **Accuracy:** Vector debates before text generation eliminate hallucinations in tool syntax.
2. **Economy:** The agent thinks inside the matrix without generating thousands of `Thought:` text tokens, radically reducing inference costs.
3. **Security:** The Latent Firewall blocks malicious intents even before the model starts vocalizing them.

**[ZH]** 与标准的 LLM 代理不同，CIDA 在隐藏状态的级别上运行（系统 2）。
1. **准确性：** 文本生成之前的向量辩论消除了工具语法中的幻觉。
2. **经济性：** 代理在矩阵内部思考，而无需生成数以千计的 `Thought:` 文本标记，从而从根本上降低了推理成本。
3. **安全性：** 潜在防火墙 (Latent Firewall) 甚至在模型开始表达恶意意图之前就将其阻止。

---

### 4. Создание мультиагентных пайплайнов | Creating Multi-Agent Pipelines | 创建多代理管道

**[RU]** Класс `CidaSwarm` позволяет связывать агентов в конвейеры (pipelines). Уникальность CIDA в том, что агенты передают друг другу Латентную Память (`LatentMemory`) — векторы, а не текст. Это предотвращает потерю контекста при долгой работе.
**[EN]** The `CidaSwarm` class allows linking agents into pipelines. CIDA's uniqueness is that agents pass Latent Memory (`LatentMemory`) to each other—vectors, not text. This prevents context loss during long operations.
**[ZH]** `CidaSwarm` 类允许将代理链接到管道中。CIDA 的独特之处在于代理相互传递潜在内存 (`LatentMemory`)——向量，而不是文本。这防止了在长时间操作中丢失上下文。

```python
from cida_plugin.inference.swarm import CidaSwarm

# [RU] Конфигурация двух агентов
# [EN] Configuration of two agents
# [ZH] 两个代理的配置
swarm_config = {
    "Researcher": {"role": "Finds data on the web", "tools": {"call_web_api": agent.call_web_api}},
    "Summarizer": {"role": "Writes concise executive summaries", "tools": {}}
}

swarm = CidaSwarm(
    wrapper=orchestrator,
    llm=orchestrator.llm,
    tokenizer=orchestrator.tokenizer,
    agent_configs=swarm_config
)

# [RU] Запуск агентов последовательно. Researcher передает векторы Summarizer'у.
# [EN] Running agents sequentially. Researcher passes vectors to Summarizer.
# [ZH] 顺序运行代理。Researcher 将向量传递给 Summarizer。
results = swarm.run(
    task="Research the latest Apple stock drop and summarize the reasons.",
    pipeline=["Researcher", "Summarizer"]
)

print(results["Summarizer"])
```

---

### 5. Масштабирование и Развертывание | Scaling and Deployment | 扩展和部署

**[RU]** Библиотека `cida_inference` спроектирована для масштабирования в Enterprise-окружениях.
* **Аппаратное обеспечение:** Оркестратор поддерживает квантование `bitsandbytes` (4-bit и 8-bit), что позволяет запускать агентов на потребительских GPU (например, RTX 3090/4090).
* **Многопоточность:** Метод `run_parallel_then_merge` в `CidaSwarm` позволяет нескольким агентам (Fan-out) одновременно собирать данные из разных источников, а затем единому агенту-синтезатору (Fan-in) объединять векторную память в финальный ответ.

**[EN]** The `cida_inference` library is designed to scale in Enterprise environments.
* **Hardware:** The orchestrator supports `bitsandbytes` quantization (4-bit and 8-bit), allowing agents to run on consumer GPUs (e.g., RTX 3090/4090).
* **Multithreading:** The `run_parallel_then_merge` method in `CidaSwarm` allows multiple agents (Fan-out) to simultaneously collect data from different sources, and then a single synthesizer agent (Fan-in) merges the vector memory into a final answer.

**[ZH]** `cida_inference` 库专为在企业环境中扩展而设计。
* **硬件：** 编排器支持 `bitsandbytes` 量化（4 位和 8 位），允许代理在消费者 GPU（例如，RTX 3090/4090）上运行。
* **多线程：** `CidaSwarm` 中的 `run_parallel_then_merge` 方法允许许多代理 (Fan-out) 同时从不同来源收集数据，然后由单个合成器代理 (Fan-in) 将向量内存合并为最终答案。

---

### 6. Ключевые плюшки библиотеки | Key Bonus Features | 关键奖励功能

**[RU]** CIDA предоставляет уникальные алгоритмы, которых нет в других фреймворках:
1. **Скрытый Режим (Stealth Mode):** Активация `silent_mode=True` полностью выключает текстовую генерацию рассуждений. Агент становится "невидимым", отвечая мгновенно и экономя ресурсы.
2. **Автономные Откаты (Latent Rollback):** Если API возвращает `Status 400` или происходит программная ошибка, агент автоматически отматывает свое состояние назад на 1 шаг, меняет аргументы вызова и пробует снова, не возвращая ошибку пользователю.
3. **Быстрый Выход (Fast Path Early Exit):** Перед началом вычислений CIDA оценивает сложность задачи. Если дисперсия внутренних мнений (`variance < 0.05`), фреймворк пропускает циклы дебатов, ускоряя генерацию ответа в 3 раза.

**[EN]** CIDA provides unique algorithms not found in other frameworks:
1. **Stealth Mode:** Activating `silent_mode=True` completely turns off textual reasoning generation. The agent becomes "invisible", responding instantly and saving resources.
2. **Latent Rollback:** If an API returns `Status 400` or a software error occurs, the agent automatically rewinds its state back by 1 step, changes the invocation arguments, and tries again without returning an error to the user.
3. **Fast Path Early Exit:** Before starting computations, CIDA evaluates task complexity. If the variance of internal opinions is low (`variance < 0.05`), the framework skips debate loops, speeding up answer generation by 3 times.

**[ZH]** CIDA 提供了其他框架中没有的独特算法：
1. **隐身模式 (Stealth Mode)：** 激活 `silent_mode=True` 将完全关闭文本推理生成。代理变得“不可见”，即时响应并节省资源。
2. **潜在回滚 (Latent Rollback)：** 如果 API 返回 `Status 400` 或发生软件错误，代理会自动将其状态后退 1 步，更改调用参数，然后重试，而不会向用户返回错误。
3. **快速路径提前退出 (Fast Path Early Exit)：** 在开始计算之前，CIDA 会评估任务复杂性。如果内部意见的方差很低 (`variance < 0.05`)，框架将跳过辩论循环，将答案生成速度提高 3 倍。
