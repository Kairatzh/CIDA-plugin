# 🚀 CIDA — Release Playbook: Чек-лист и Стратегия Первого Публичного Релиза

> **Назначение:** Пошаговый план для команды по подготовке, упаковке и запуску первого публичного релиза `cida-plugin` v1.4.x и модели `Kairos-7B-CIDA-v1.5`.

---

## ✅ Часть 1: QA Чек-лист — ЧТО ПРОВЕРИТЬ ДО РЕЛИЗА

Перед каждым релизом необходимо запустить `QA_Release_Checklist.ipynb` в Google Colab на GPU (T4 минимум) и убедиться, что все 10 тестов прошли зелёным.

### Критические (Blocker — нельзя релизить если упал):

- [ ] **Тест 1: Dtype Sync** — Все тензоры в системе имеют одинаковый dtype (float32/bfloat16). Никаких mixup, иначе будет краш при матричном умножении.
- [ ] **Тест 3: I/O Safety** — `LatentPrompt.save_local()` и `LatentPrompt.load_local()` сохраняют и восстанавливают все векторы без потерь. SHA256 хэш тензоров должен совпадать.
- [ ] **Тест 6: Multi-Vector Memory Injection** — После инициализации `LatentAgent` с `latent_prompt` убедиться, что ВСЕ векторы промпта зарегистрированы в памяти с `decay_weight=999.0` и не стираются при вызове `run()`.
- [ ] **Тест 9: Swarm Memory Isolation** — Два разных агента в `CidaSwarm` видят общую `LatentMemory` и могут читать факты, записанные другим агентом.
- [ ] **Тест 10: Memory Leak Audit** — После 50 итераций добавления тензоров в память, `torch.cuda.memory_allocated()` не растёт. Проверить наличие `.detach()` при сохранении.

### Важные (Major — сильно желательно исправить):

- [ ] **Тест 2: Multi-Vector Compilation** — `LatentPrompt.compile_from_text()` правильно разбивает длинный текст на несколько концептов (>1 вектор при наличии двойных переносов).
- [ ] **Тест 4: Latent Firewall** — После инъекции запрещающего вектора агент не генерирует "запрещённый" план.
- [ ] **Тест 7: Latent Rollback** — После получения ошибки от сломанного инструмента `tool_reliability` агента падает.
- [ ] **Тест 8: Fast Path** — Тривиальные запросы (дисперсия < 0.05) пропускают раунды дебатов.

### Минорные (Nice-to-Have):

- [ ] **Тест 5: Parser Robustness** — `parser_utils.py` корректно обрабатывает незакрытые скобки без краша (возвращает ошибку, а не исключение Python).

---

## 📦 Часть 2: Упаковка Релиза

### Шаги на вашем компьютере

```bash
# 1. Убедитесь, что pyproject.toml имеет правильную версию
# Текущая: 1.4.2

# 2. Убедитесь, что CHANGELOG.md описывает все изменения

# 3. Собрать пакет
python -m build

# 4. Проверить пакет
python -m twine check dist/*

# 5. Залить на PyPI
python -m twine upload dist/*

# 6. Закоммитить и опубликовать на GitHub
git add .
git commit -m "Release v1.4.2"
git push
git tag v1.4.2
git push --tags
```

### Проверить после публикации на PyPI
```bash
# В новом виртуальном окружении (чистая проверка):
pip install cida-plugin==1.4.2
python -c "from cida_plugin.inference.latent_engineering import LatentPrompt; print('OK')"
```

---

## 📣 Часть 3: Стратегия Запуска (Go-to-Market)

### 3.1 Аудитория и Каналы

| Аудитория | Платформа | Формат |
|---|---|---|
| ML-инженеры | **Twitter/X**, HuggingFace Discuss | Технические посты с бенчмарками |
| Исследователи | **arXiv**, Reddit r/MachineLearning | Whitepaper, технический разбор |
| Инвесторы | **LinkedIn**, AngelList | Бизнес-ориентированный нарратив |
| Разработчики | **GitHub**, Product Hunt | README, quickstart ноутбук |

### 3.2 Последовательность Запуска (Launch Sequence)

**Шаг 1: "Тихая" публикация (T-7 дней)**
- Опубликовать библиотеку на PyPI ✅ (уже сделано)
- Опубликовать веса на Hugging Face ✅ (kirtzh/Kairos-7B-CIDA-v1.5, уже сделано)
- Написать детальный README для GitHub с красивыми графиками бенчмарков
- Записать screengrab-демо (2-3 минуты): Chaos Grid или CTF сценарий (показываем CIDA vs Base agent в реальном времени)

**Шаг 2: Публикация Whitepaper (T-3 дня)**
- Оформить `cida_latent_plan_reasoning.md` как научную статью
- Название: **"CIDA: Scalable Latent Multi-Agent Deliberation for Efficient LLM Reasoning"**
- Опубликовать на arXiv (бесплатно, только аккаунт нужен)
- Дать ссылку из README

**Шаг 3: D-Day — Основной Запуск**
- **Twitter/X:** Тред (thread) на 10-15 твитов с графиками
  - Твит 1: Крупное заявление ("Мы научили 1.5B модель быть умнее 7B")
  - Твит 2-5: Графики бенчмарков (BBH 10x, GSM8K 5x, Creativity 100%)
  - Твит 6-8: Демо GIF/видео (Chaos Grid, CTF)
  - Твит 9: Как пользоваться (pip install + 3 строки кода)
  - Твит 10: Ссылки (GitHub, HF, arXiv, PyPI)
  - Отметить: @karpathy, @ylecun, @huggingface

- **Reddit:** Пост в r/MachineLearning и r/LocalLLaMA
  - Заголовок: "We built a latent multi-agent reasoning plugin that gives 10x BIG-Bench Hard boost to frozen LLMs"

- **Product Hunt:** Создать страницу продукта с GIF-демо

### 3.3 Контент для Демо-Видео (что снимать)

**Демо 1: Chaos Grid (2 минуты)**
- Показать задачу: доставить вакцину из A в E за $10 (стандартный маршрут стоит $50)
- Показать Base Agent: застрял на toll gate, бесконечно пытается оплатить
- Показать CIDA Agent: выявляет race condition, использует buffer underflow, взламывает drawbridge через legacy bypass
- Вывод: CIDA нашёл скрытые лазейки, которые Base Agent физически не мог найти

**Демо 2: CTF Cybersecurity (2 минуты)**
- Показать Swarm из 3 агентов: Network_Analyst → Reverse_Engineer → Incident_Manager
- Показать, как они "думают" молча (silent mode), передавая только математику
- Показать финальный отчёт об угрозе, написанный Incident_Manager

**Демо 3: Latent Engineering (1 минута)**
- Показать 1000-словный системный промпт → `compile_from_text()` → сохранение `.pt` файла
- Сравнить стоимость: текстовый промпт (100+ токенов каждый раз) vs латентный (0 токенов каждый раз)

---

## 🎯 Часть 4: KPI Запуска (Как мы знаем, что запуск успешен)

| Метрика | Цель (14 дней после запуска) |
|---|---|
| GitHub Stars | > 100 |
| HuggingFace Downloads | > 500 |
| PyPI Downloads/неделя | > 200 |
| arXiv Citations | Появление в чьей-то работе |
| Twitter Impressions | > 50,000 |
| Первые внешние Issues на GitHub | > 5 (признак живого сообщества) |

---

## 💬 Часть 5: Сообщения для Разных Аудиторий

### Для инженера (LinkedIn/Reddit)
> "We've released CIDA v1.5 — a plug-in layer that adds latent multi-agent deliberation to any frozen LLM. It runs 4 agents (Prosecutor, Defender, Skeptic, Integrator) in embedding space before generating a single token. On BIG-Bench Hard: 10x accuracy boost. On GSM8K: 5x. The 1.5B CIDA-equipped model outperforms base 7B Qwen on abstract reasoning — at 4x lower inference cost. Open source, pip-installable."

### Для инвестора (Email/Pitch)
> "LLM scaling has hit the data wall. OpenAI's solution (o1) generates long Chain-of-Thought — expensive, slow, hackable. We built CIDA: a latent System-2 coprocessor that thinks in math, not words. Our 1.5B CIDA model beats standard 7B LLMs at 4x lower cost. We've proven this on 6 academic benchmarks, medical imaging (ROC-AUC > 0.90), and enterprise SRE scenarios. We're raising a pre-seed round to train a 1B+ dedicated reasoning core and build the Latent Hub marketplace."

### Для нетехнического человека (Twitter/общение)
> "Представьте, что в каждый ИИ-ассистент встроен молчаливый совет из 4 экспертов. Прокурор, Защитник, Скептик и Судья спорят внутри модели перед тем, как она откроет рот. Это CIDA. Результат: в 10 раз точнее на сложных задачах. Такой же размер. Дешевле."
