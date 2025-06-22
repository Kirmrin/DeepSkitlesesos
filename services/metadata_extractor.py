from langsmith import traceable
from gigachain import GigaChatModel
import logging
import config
import re
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class MetadataExtractor:
    def __init__(self):
        self.logger = logging.getLogger("metadata_extractor")
        self.llm = GigaChatModel(
            model="GigaChat-Plus",
            temperature=0.3,
            api_key=config.GIGACHAT_API_KEY,
            max_tokens=500
        )
        self.entity_patterns = self._load_entity_patterns()
        self.cache = {}
        
    @traceable
    def extract_metadata(self, user_query: str, user_context: Dict) -> Dict:
        """
        Извлекает структурированные метаданные из запроса
        :param user_query: Текст запроса пользователя
        :param user_context: Контекст пользователя (роль, история и т.д.)
        :return: Словарь с метаданными
        """
        # Проверка кеша
        cache_key = f"{user_query}-{user_context.get('user_id','')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Быстрое извлечение с помощью правил
            quick_metadata = self._quick_extraction(user_query)
            
            # Углубленный анализ через LLM при необходимости
            if self._needs_deep_analysis(user_query, quick_metadata):
                metadata = self._deep_analysis(user_query, user_context)
            else:
                metadata = quick_metadata
                
            # Обогащение контекстными данными
            metadata.update({
                "extraction_method": "deep" if "deep" in metadata else "quick",
                "user_role": user_context.get("role", "unknown"),
                "timestamp": datetime.now().isoformat()
            })
            
            # Кеширование результата
            self.cache[cache_key] = metadata
            return metadata
            
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {str(e)}")
            return self._fallback_metadata(user_query)

    def _quick_extraction(self, query: str) -> Dict:
        """Быстрое извлечение метаданных с помощью правил"""
        metadata = {
            "entities": [],
            "intent": "unknown",
            "parameters": {},
            "content_type": "text"
        }
        
        # Извлечение дат
        date_matches = re.findall(r"\b\d{2}\.\d{2}\.\d{4}\b|\b\d{4}-\d{2}-\d{2}\b", query)
        if date_matches:
            metadata["parameters"]["dates"] = date_matches
            metadata["entities"].append("date")
        
        # Извлечение чисел
        number_matches = re.findall(r"\b\d+\b", query)
        if number_matches:
            metadata["parameters"]["numbers"] = list(map(int, number_matches))
            metadata["entities"].append("number")
        
        # Определение типа контента
        if any(word in query.lower() for word in ["график", "диаграм", "визуализ"]):
            metadata["content_type"] = "chart"
        elif any(word in query.lower() for word in ["таблиц", "список", "перечень"]):
            metadata["content_type"] = "table"
        elif any(word in query.lower() for word in ["отчет", "документ", "файл"]):
            metadata["content_type"] = "report"
        
        # Определение намерения
        if any(word in query.lower() for word in ["покажи", "выведи", "отобраз"]):
            metadata["intent"] = "show_data"
        elif any(word in query.lower() for word in ["сравни", "против", "vs"]):
            metadata["intent"] = "compare"
        elif any(word in query.lower() for word in ["анализ", "тренд", "динамик"]):
            metadata["intent"] = "analyze"
        elif any(word in query.lower() for word in ["почему", "причина", "объясни"]):
            metadata["intent"] = "explain"
            
        return metadata

    def _needs_deep_analysis(self, query: str, quick_metadata: Dict) -> bool:
        """Определяет, нужен ли углубленный анализ"""
        # Если намерение не определено
        if quick_metadata["intent"] == "unknown":
            return True
            
        # Если есть числа, но нет контекста
        if "numbers" in quick_metadata["parameters"] and not quick_metadata["entities"]:
            return True
            
        # Если запрос длинный (более 7 слов)
        if len(query.split()) > 7:
            return True
            
        return False

    def _deep_analysis(self, query: str, context: Dict) -> Dict:
        """Углубленный анализ запроса с помощью LLM"""
        prompt = f"""Ты эксперт по извлечению метаданных. Проанализируй запрос пользователя:
        
### Запрос:
{query}

### Контекст:
- Роль пользователя: {context.get('role', 'unknown')}
- Текущая дата: {datetime.now().strftime('%Y-%m-%d')}

### Задача:
Извлеки структурированные метаданные в формате JSON со следующими полями:
1. entities: список сущностей (например, ["дата", "продукт", "регион"])
2. intent: основное намерение (show_data, compare, analyze, explain, request_help)
3. parameters: словарь параметров ({{"date_range": ["2023-01-01", "2023-12-31"], "metrics": ["продажи"]}})
4. content_type: тип ожидаемого результата (text, table, chart, report)
5. urgency: срочность запроса (low, medium, high)

### Пример:
Запрос: "Покажи продажи телефонов в Москве за последний месяц"
Ответ: {{
  "entities": ["продукт", "регион", "дата"],
  "intent": "show_data",
  "parameters": {{"product": "телефоны", "region": "Москва", "period": "last_month"}},
  "content_type": "table",
  "urgency": "medium"
}}
"""
        response = self.llm.generate(prompt, response_format="json")
        metadata = json.loads(response)
        
        # Добавляем флаг глубокого анализа
        metadata["deep_analysis"] = True
        return metadata

    def _load_entity_patterns(self) -> Dict[str, List[str]]:
        """Загружает паттерны для распознавания сущностей"""
        return {
            "date": [
                r"\b\d{2}\.\d{2}\.\d{4}\b",  # DD.MM.YYYY
                r"\b\d{4}-\d{2}-\d{2}\b",     # YYYY-MM-DD
                r"\b(?:квартал|месяц|недел[яю]|год)[а-я]*\b",
                r"\bпоследн\w+ \d+ (?:дн[ея]|месяц|недел)\b"
            ],
            "product": [
                r"\bтелефон\w*\b",
                r"\bноутбук\w*\b",
                r"\bпродукт\w*\b",
                r"\bтовар\w*\b"
            ],
            "region": [
                r"\bМоскв\w*\b",
                r"\bСанкт-Петербург\w*\b",
                r"\bрегион\w*\b",
                r"\bгород\w*\b"
            ],
            "metric": [
                r"\bпродаж\w*\b",
                r"\bдоход\w*\b",
                r"\bприбыл\w*\b",
                r"\bконверси\w*\b"
            ]
        }

    def _fallback_metadata(self, query: str) -> Dict:
        """Фолбэк при ошибках извлечения"""
        return {
            "entities": [],
            "intent": "unknown",
            "parameters": {},
            "content_type": "text",
            "urgency": "medium",
            "error": "extraction_failed"
        }

    def extract_from_history(self, history: List[Dict]) -> Dict:
        """Извлекает метаданные из истории диалога"""
        last_three = [msg["content"] for msg in history[-3:]]
        context = " | ".join(last_three)
        
        prompt = f"""Проанализируй историю диалога и извлеки контекстные метаданные:
        
История:
{context}

Сформируй JSON с полями:
- main_topic: основная тема диалога
- pending_actions: список невыполненных действий
- user_preferences: предпочтения пользователя
"""
        response = self.llm.generate(prompt, response_format="json")
        return json.loads(response)

    def update_entity_patterns(self, entity_type: str, patterns: List[str]):
        """Обновляет паттерны для распознавания сущностей"""
        if entity_type not in self.entity_patterns:
            self.entity_patterns[entity_type] = []
            
        self.entity_patterns[entity_type].extend(patterns)
        self.logger.info(f"Updated patterns for {entity_type}")

    def clear_cache(self):
        """Очищает кеш извлечения"""
        self.cache = {}
        self.logger.info("Metadata cache cleared")