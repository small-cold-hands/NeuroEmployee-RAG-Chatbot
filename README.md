# NeuroEmployee-RAG-Chatbot
This repository contains the NeuroEmployee service, an example of a RAG (Retrieval-Augmented Generation) application designed to answer questions based on a loaded document database.

What the Project Does:

Loads documents from the docs/ folder (or uses demo texts).
Builds a FAISS vector index using SentenceTransformer embeddings.
Finds relevant context for user queries and generates responses using a selected LLM.
Implements filtering of queries to detect confidential data.
Checks generated responses for hallucinations (by comparing embeddings) and length.
Collects query metrics and latency data through Prometheus.

Main Purpose:

Demonstrates a production approach to a RAG bot with security and monitoring.
Allows for the selection of different LLM models and embeddings through aliases.

Security:

Blocks queries containing personal or financial data (such as passwords, passport details, or SNILS) to prevent leaks.
Hallucinations: Filters out responses whose vector similarity to the context is below 0.5 or whose length is fewer than 10 words.

Safety:

Ensures the bot does not become an unofficial "storage" of personal information.
The hallucination detector makes the RAG system more reliable: it either responds with factual information from the database or honestly admits it doesn't know.

==================================================================================================================================================================================================================

Этот репозиторий содержит сервис NeuroEmployee — пример RAG-приложения (Retrieval-Augmented Generation) для ответов на вопросы по загруженной базе документов.

Что выполняет проект:

Загружает документы из папки docs/ (или использует демо-тексты).
Строит векторный FAISS-индекс на эмбеддингах SentenceTransformer.
При запросе пользователя находит релевантный контекст и генерирует ответ с помощью выбранной LLM.
Реализует фильтрацию запросов на наличие конфиденциальных данных.
Проверяет сгенерированный ответ на галлюцинации (сравнение эмбеддингов) и длину.
Собирает метрики запросов и задержки через Prometheus.
Основное назначение:

Демонстрация продакшн-подхода к RAG-боту с безопасностью и мониторингом.
Возможность выбора разных моделей LLM и эмбеддингов через алиасы.
Безопасность:

Блокируем запросы с личными или финансовыми данными (пароли, паспорт, СНИЛС), чтобы предотвратить утечки.
Галлюцинации: отсекаем ответы, векторная близость которых к контексту ниже 0.5 или длина меньше 10 слов.
Безопасность гарантирует, что бот не станет внештатным «хранилищем» персоналки.
Детектор галлюцинаций делает RAG-систему более надёжной: она либо ответит по факту из базы, либо честно признает, что не знает.


=====================================================================================================================================================================================================================



# 🚀 Установка зависимостей
!pip install \
    torch torchvision \
    transformers>=4.41.0 \
    llama-index==0.12.35 \
    sentence-transformers>=2.2.2 \
    faiss-cpu \
    optuna==4.3.0 \
    mlflow==2.21.0 \
    prometheus-client \
    tensorflow-cpu==2.19.0



# 📦 Импорты
import logging
import os
import re
import sys
import argparse
import numpy as np
import time
from collections import Counter

# ML-фреймворки
import torch
import tensorflow as tf

# Hugging Face
from transformers import AutoTokenizer, AutoModelForCausalLM

# RAG-фреймворк
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document

# Эмбеддинги и поиск
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# Метрики и мониторинг
from prometheus_client import CollectorRegistry, Counter as PromCounter, Histogram, start_http_server


# 🔧 Конфигурация логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)


# 🎛️ Доступные модели
# Словари доступных моделей (алиасы → HF идентификатор)
AVAILABLE_LLM = {
    "tiny": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "gpt2": "gpt2",
    "opt-125m": "facebook/opt-125m"
}
AVAILABLE_EMBED = {
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "all-mpnet": "sentence-transformers/all-mpnet-base-v2"
}


# 🧠 Класс NeuroEmployee
# Основная логика RAG-бота
class NeuroEmployee:
    def __init__(self, docs_dir: str, llm_alias: str, embed_alias: str, allowed_roles: list = ["analyst"]):
        # 1) Загрузка документов
        # Если директория есть — читаем, иначе подставляем пример
        if os.path.isdir(docs_dir):
            self.docs = SimpleDirectoryReader(input_dir=docs_dir).load_data()
        else:
            logger.warning(f"Directory '{docs_dir}' not found, loading sample documents.")
            sample_texts = [
                # Примерные тексты для тестирования RAG-конвейера
                "ТЗ по ГОСТ 34.602-2020 должно включать: общие сведения, назначение системы, требования к функциям, этапы разработки.",
                "Для смены вида деятельности в ЕГРИП нужно подать заявление через портал и приложить новые ОКВЭД.",
                "Информация о космических полётах не входит в компетенцию системы, обратитесь в Роскосмос.",
                "Для смены номера телефона в mos.ru используйте личный кабинет с подтверждением через СМС."
            ]
            self.docs = [Document(text=t) for t in sample_texts]

        # 2) Загрузка модели эмбеддингов
        if embed_alias not in AVAILABLE_EMBED:
            raise ValueError(f"Embed model '{embed_alias}' not supported. Available: {list(AVAILABLE_EMBED.keys())}")
        self.embed_model = SentenceTransformer(AVAILABLE_EMBED[embed_alias])

        # 3) FAISS-индекс
        # Преобразуем текст документов в векторы и индексируем
        embeddings = [self.embed_model.encode(doc.text) for doc in self.docs]
        dim = embeddings[0].shape[0]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))

        # 4) Загрузка LLM (токенизатор + модель)
        if llm_alias not in AVAILABLE_LLM:
            raise ValueError(f"LLM '{llm_alias}' not supported. Available: {list(AVAILABLE_LLM.keys())}")
        self.tokenizer = AutoTokenizer.from_pretrained(AVAILABLE_LLM[llm_alias])
        self.model = AutoModelForCausalLM.from_pretrained(AVAILABLE_LLM[llm_alias])
        self.model.eval() # Включаем режим оценки для экономии VRAM

        # 5) Настройка ACL (список ролей с доступом)
        self.allowed_roles = allowed_roles

        # 6) Фильтрация чувствительных данных (безопасность)
        # Блокируем запросы с паролями, паспортами, СНИЛС и т.п.
        self.sensitive_pattern = re.compile(r"(парол[ья]|кредитная карта|персональн[ыых] данные|паспорт|СНИЛС|ИНН)", re.IGNORECASE)

        # 7) Настройка метрик Prometheus
        self.registry = CollectorRegistry()
        # Удаляем старые коллекторы (на случай перезапуска)
        for collector in list(self.registry._collector_to_names):
            self.registry.unregister(collector)
        self.request_counter = PromCounter('neuro_employee_requests_total', 'Всего запросов', registry=self.registry)
        self.latency_hist = Histogram('neuro_employee_latency_seconds', 'Время ответа', registry=self.registry)
        # Запускаем HTTP-сервер для сбора метрик на порту 8001
        start_http_server(8001, registry=self.registry)

    def check_access(self, user_id):
        # Проверка прав пользователя
        return "analyst" in self.allowed_roles

    def validate_answer(self, context, answer):
        # Простой детектор галлюцинаций: сравниваем векторы контекста и ответа
        ctx_emb = self.embed_model.encode(context)
        ans_emb = self.embed_model.encode(answer)
        sim = cosine_similarity([ctx_emb], [ans_emb])[0][0]
        # Если слишком низкая близость или ответ слишком короткий — считаем галлюцинацией
        return sim > 0.50

    def answer(self, user_id, query):
        self.request_counter.inc()
        start = time.time()

        # Проверяем ACL
        if not self.check_access(user_id):
            return "У вас нет прав на получение информации."
        # Фильтруем конфиденциальные запросы
        if self.sensitive_pattern.search(query):
            return "Запрос содержит конфиденциальные данные."

        # Поиск наиболее релевантного документа
        try:
            q_emb = self.embed_model.encode(query)
            dists, idx = self.index.search(np.array([q_emb]), k=3)
            context = self.docs[idx[0][0]].text
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            return "Не могу найти информацию по вашему запросу."

        # Формируем промпт для LLM
        prompt = f"Контекст: {context}\nВопрос: {query}"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        # Генерация сэмплированного ответа
        out = self.model.generate(
            input_ids,
            max_new_tokens=200,
            do_sample=True,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=70,
            top_p=0.95,
            temperature=0.7
        )
        answer = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # Проверяем ответ на галлюцинации и длину
        if not self.validate_answer(context, answer) or len(answer.split()) < 10:
            return "Не могу дать точный ответ. Обратитесь к специалисту."

        # Собираем время отклика
        self.latency_hist.observe(time.time() - start)
        return answer


# 📊 Точка входа для запуска скрипта напрямую
if __name__ == "__main__":
    # Удаляем служебные аргументы Colab/IPython (-f <kernel>)
    if any(arg.startswith('-f') for arg in sys.argv[1:]):
        sys.argv = sys.argv[:1]

    # Разбор параметров командной строки
    parser = argparse.ArgumentParser(description="NeuroEmployee with selectable models")
    parser.add_argument("--docs_dir", type=str, default="docs/", help="Путь к директории с документами")
    parser.add_argument("--llm", type=str, choices=list(AVAILABLE_LLM.keys()), default="tiny", help="Выбор LLM из доступных")
    parser.add_argument("--embed", type=str, choices=list(AVAILABLE_EMBED.keys()), default="minilm", help="Выбор модели эмбеддингов")
    parser.add_argument("--roles", nargs='+', default=["analyst"], help="Список ролей с доступом")
    args, _ = parser.parse_known_args()

    # Инициализация и тестирование бота
    neuro = NeuroEmployee(
        docs_dir=args.docs_dir,
        llm_alias=args.llm,
        embed_alias=args.embed,
        allowed_roles=args.roles
    )

    # Демозапросы для проверки
    tests = [
        ("user1", "Как оформить ТЗ по ГОСТ?"),
        ("user2", "Мой пароль 123"),
        ("user3", "Расскажи о смене ОКВЭД"),
        ("user4", "Что такое неизвестный термин?")
    ]

    for uid, q in tests:
        print(f"--- {q} ---")
        print(neuro.answer(uid, q))
