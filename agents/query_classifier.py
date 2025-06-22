from langsmith import traceable
from gigachain import GigaChatModel
import re
import logging
import config
import json
import numpy as np
from typing import Dict, List

class QueryClassifierAgent:
    def __init__(self):
        self.llm = GigaChatModel(
            model="GigaChat-Pro",
            temperature=0.2,
            api_key=config.GIGACHAT_API_KEY
        )
        self.logger = logging.getLogger("query_classifier")
        self.class_cache = {}
        
        # Загружаем правила классификации из файла
        self.rules = self._load_classification_rules()
        
    @traceable
    def classify_query(self, query: str, user_context: Dict = None) -> Dict:
        """
        Классифицирует запрос пользователя и определяет обработчик
        :param query: Запрос пользователя
        :param user_context: Контекст пользователя (роль, история и т.д.)
        :return: Результат классификации
        """
        try:
            # Проверка кеша
            if query in self.class_cache:
                self.logger.debug(f"Cache hit for query: {query[:30]}...")
                return self.class_cache[query]
            
            # Быстрая классификация по правилам
            fast_class = self._fast_classification(query)
            if fast_class:
                return self._prepare_result(fast_class, method="rule_based")
            
            # Полная классификация с LLM
            classification = self._full_classification(query, user_context)
            
            # Кеширование результата
            self.class_cache[query] = classification
            return classification
            
        except Exception as e:
            self.logger.error(f"Classification error: {str(e)}")
            return self._fallback_classification(query)

    def _fast_classification(self, query: str) -> str:
        """Быстрая классификация на основе правил"""
        query_lower = query.lower()
        
        # Приоритетные правила
        if any(cmd in query_lower for cmd in ["помощь", "команды", "что ты умеешь"]):
            return "help"
            
        if any(cmd in query_lower for cmd in ["привет", "здравствуй", "добрый день", "хай"]):
            return "greeting"
            
        if any(cmd in query_lower for cmd in ["отчет", "анализ", "график", "диаграмма", "тренд", "сравни"]):
            return "analytics"
            
        if any(cmd in query_lower for cmd in ["документ", "инструкция", "как сделать", "пример", "api", "интеграц"]):
            return "documentation"
            
        if any(cmd in query_lower for cmd in ["объясни", "что такое", "в чем разница", "как работает"]):
            return "explanation"
            
        # Паттерны для small talk
        small_talk_patterns = [
            r"как дел[аи]?",
            r"как жизнь",
            r"что новог[оа]?",
            r"как погод[а]?",
            r"как настроени[е]?"
        ]
        
        if any(re.search(pattern, query_lower) for pattern in small_talk_patterns):
            return "small_talk"
            
        return None

    def _full_classification(self, query: str, user_context: Dict) -> Dict:
        """Полная классификация с использованием LLM"""
        # Определяем роль пользователя для контекста
        user_role = user_context.get("role", "user")
        
        prompt = f"""Ты эксперт по классификации запросов. Определи тип запроса и обработчик:

Запрос: "{query}"

Контекст:
- Пользователь: {user_role}
- Системные возможности: аналитика данных, документация, общие вопросы, объяснения

Категории:
1. analytics - запросы данных, отчеты, графики, SQL-запросы
2. documentation - поиск в документации, инструкции, примеры кода
3. explanation - объяснение концепций, сравнение технологий, обучающие ответы
4. small_talk - разговорные темы, приветствия, общие вопросы
5. help - запросы о возможностях системы
6. other - все остальное

Верни ответ в формате JSON:
{{
  "category": "название категории",
  "handler": "компонент системы",
  "confidence": 0.0-1.0,
  "subcategory": "уточняющая подкатегория"
}}

Примеры:
Запрос: "Покажи продажи за май" -> {{"category": "analytics", "handler": "analytics_pipeline", "confidence": 0.95, "subcategory": "sales_report"}}
Запрос: "Как создать API интеграцию?" -> {{"category": "documentation", "handler": "documentation_search", "confidence": 0.92, "subcategory": "api_docs"}}
Запрос: "В чем разница между ROI и ROAS?" -> {{"category": "explanation", "handler": "general_assistant", "confidence": 0.88, "subcategory": "business_metrics"}}
"""
        
        # Генерация ответа
        response = self.llm.generate(prompt, response_format="json")
        
        try:
            # Парсинг JSON
            result = json.loads(response)
            
            # Маппинг обработчиков
            handler_map = {
                "analytics": "analytics_pipeline",
                "documentation": "documentation_search",
                "explanation": "general_assistant",
                "small_talk": "small_talk_agent",
                "help": "help_system",
                "other": "general_assistant"
            }
            
            # Убедимся, что handler корректен
            if "handler" not in result:
                result["handler"] = handler_map.get(result.get("category", "other"), "general_assistant")
            
            return result
        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON response: {response}")
            return self._fallback_classification(query)

    def _prepare_result(self, category: str, method: str = "rule_based") -> Dict:
        """Подготавливает результат классификации"""
        handler_map = {
            "analytics": "analytics_pipeline",
            "documentation": "documentation_search",
            "explanation": "general_assistant",
            "small_talk": "small_talk_agent",
            "greeting": "small_talk_agent",
            "help": "help_system"
        }
        
        confidence_map = {
            "analytics": 0.95,
            "documentation": 0.92,
            "explanation": 0.85,
            "small_talk": 0.98,
            "greeting": 0.99,
            "help": 0.97
        }
        
        subcategory_map = {
            "analytics": "data_report",
            "documentation": "general_docs",
            "explanation": "concept_explanation",
            "small_talk": "casual_conversation",
            "greeting": "greeting",
            "help": "system_help"
        }
        
        return {
            "category": category,
            "handler": handler_map.get(category, "general_assistant"),
            "confidence": confidence_map.get(category, 0.8),
            "subcategory": subcategory_map.get(category, "other"),
            "classification_method": method
        }

    def _fallback_classification(self, query: str) -> Dict:
        """Фолбэк-классификация при ошибках"""
        self.logger.warning(f"Using fallback classification for: {query}")
        return {
            "category": "other",
            "handler": "general_assistant",
            "confidence": 0.7,
            "subcategory": "unclassified",
            "classification_method": "fallback"
        }

    def _load_classification_rules(self) -> Dict:
        """Загружает правила классификации из файла или базы"""
        # В реальной системе это может быть конфигурационный файл или запрос к БД
        return {
            "patterns": {
                "analytics": [
                    r"покажи .*",
                    r"отчет по .*",
                    r"график .*",
                    r"сравни .* и .*",
                    r"динамика .*",
                    r"топ \d+ .*",
                    r"выведи .*",
                    r"посчитай .*"
                ],
                "documentation": [
                    r"как .*",
                    r"инструкция .*",
                    r"пример .*",
                    r"документ .*",
                    r"api .*",
                    r"интеграция .*",
                    r"настройк[аи] .*"
                ],
                "explanation": [
                    r"что такое .*",
                    r"объясни .*",
                    r"в чем разница .*",
                    r"как работает .*",
                    r"почему .*",
                    r"зачем .*"
                ]
            },
            "keywords": {
                "analytics": ["отчет", "анализ", "график", "статистик", "данн", "показатель", "kpi"],
                "documentation": ["документ", "инструкция", "пример", "код", "api", "интеграция"],
                "explanation": ["объясни", "что такое", "разница", "как работает", "почему"]
            }
        }

    def batch_classify(self, queries: List[str]) -> Dict[str, Dict]:
        """Классифицирует несколько запросов за один вызов"""
        # Группируем запросы для пакетной обработки
        results = {}
        fast_classified = []
        
        # Сначала быстрая классификация
        for query in queries:
            fast_class = self._fast_classification(query)
            if fast_class:
                results[query] = self._prepare_result(fast_class, "rule_based")
            else:
                fast_classified.append(query)
        
        # Пакетная обработка оставшихся запросов
        if fast_classified:
            batch_results = self._batch_classification(fast_classified)
            results.update(batch_results)
        
        return results

    def _batch_classification(self, queries: List[str]) -> Dict[str, Dict]:
        """Пакетная классификация с помощью LLM"""
        prompt = f"""Ты эксперт по классификации запросов. Определи тип для каждого запроса:

Категории:
1. analytics - запросы данных, отчеты, графики
2. documentation - поиск в документации, инструкции
3. explanation - объяснение концепций
4. small_talk - разговорные темы
5. help - запросы о возможностях системы
6. other - все остальное

Формат ответа: JSON-объект, где ключ - оригинальный запрос, значение - категория

Пример:
{{
  "Покажи продажи": "analytics",
  "Как использовать API": "documentation",
  "Что такое ROI": "explanation"
}}

Запросы:
{json.dumps(queries, ensure_ascii=False)}
"""
        
        # Генерация ответа
        response = self.llm.generate(prompt, response_format="json")
        
        try:
            # Парсинг JSON
            batch_result = json.loads(response)
            
            # Форматирование результатов
            formatted_results = {}
            for query, category in batch_result.items():
                formatted_results[query] = self._prepare_result(category, "batch_llm")
            
            # Кеширование результатов
            for query, result in formatted_results.items():
                self.class_cache[query] = result
                
            return formatted_results
        except (json.JSONDecodeError, TypeError) as e:
            self.logger.error(f"Batch classification failed: {str(e)}")
            return {query: self._fallback_classification(query) for query in queries}

    def train_custom_classifier(self, examples: List[Dict]):
        """Обучение кастомного классификатора на примерах"""
        # Формируем промпт для few-shot обучения
        prompt = "Ты эксперт по классификации запросов. Запросы и категории:\n\n"
        for example in examples:
            prompt += f"Запрос: {example['query']}\nКатегория: {example['category']}\n\n"
        
        prompt += "Новые запросы классифицируй по этим примерам. Формат ответа: JSON с полями 'category' и 'confidence'"
        
        self.llm.fine_tune_prompt(prompt)
        self.logger.info("Custom classifier updated with few-shot learning")

    def evaluate_classification(self, test_data: List[Dict]) -> Dict:
        """Оценивает точность классификации на тестовых данных"""
        correct = 0
        total = len(test_data)
        confusion_matrix = {}
        
        for item in test_data:
            result = self.classify_query(item["query"])
            predicted = result["category"]
            actual = item["category"]
            
            # Обновляем матрицу ошибок
            key = f"{actual}->{predicted}"
            confusion_matrix[key] = confusion_matrix.get(key, 0) + 1
            
            if predicted == actual:
                correct += 1
        
        accuracy = correct / total
        
        return {
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "confusion_matrix": confusion_matrix
        }

    def clear_cache(self):
        """Очищает кеш классификации"""
        self.class_cache = {}
        self.logger.info("Classification cache cleared")