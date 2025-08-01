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

В этом репозитории содержится сервис NeuroEmployee — пример RAG-приложения (Retrieval-Augmented Generation), предназначенного для ответов на вопросы на основе загруженной базы документов.
Что выполняет проект:
•	Загружает документы из папки docs/ (или использует демонстрационные тексты).
•	Строит векторный индекс FAISS с использованием эмбеддингов SentenceTransformer.
•	Находит релевантный контекст для запросов пользователей и генерирует ответы с помощью выбранной модели LLM.
•	Реализует фильтрацию запросов для обнаружения конфиденциальных данных.
•	Проверяет сгенерированные ответы на наличие галлюцинаций (путём сравнения эмбеддингов) и длину.
•	Собирает метрики запросов и данные о задержках через Prometheus.
Основное назначение:
•	Демонстрация производственного подхода к RAG-боту с обеспечением безопасности и мониторинга.
•	Возможность выбора различных моделей LLM и эмбеддингов через алиасы.
Безопасность:
•	Блокирует запросы, содержащие личные или финансовые данные (например, пароли, данные паспорта или СНИЛС), чтобы предотвратить утечки.
•	Галлюцинации: отсеивает ответы, векторное сходство которых с контекстом ниже 0.5 или длина которых меньше 10 слов.
Надёжность:
•	Обеспечивает, чтобы бот не стал неофициальным "хранилищем" личной информации.
•	Детектор галлюцинаций делает RAG-систему более надёжной: она либо отвечает фактической информацией из базы, либо честно признаёт, что не знает ответа.
