import requests
import json
from config import Config
from services.monitoring import tracer
from .utils import clean_html_tags, filter_domains
import logging
from typing import List, Dict, Any
import re
import time

# Настройка логгера
logger = logging.getLogger(__name__)

class WebSearch:
    def __init__(self):
        self.api_key = Config.DUCKDUCKGO_API_KEY
        self.allowed_domains = Config.WEB_SEARCH_ALLOWED_DOMAINS
        self.max_results = Config.WEB_SEARCH_MAX_RESULTS
        self.timeout = Config.WEB_SEARCH_TIMEOUT
        self.cache = {}
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Минимальный интервал между запросами (сек)

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Выполняет веб-поиск по запросу с фильтрацией по разрешенным доменам
        
        :param query: Поисковый запрос
        :return: Список результатов в формате [{"title": "...", "url": "...", "snippet": "..."}]
        """
        # Проверка кеша
        cache_key = f"websearch:{query}"
        if cache_key in self.cache:
            logger.debug(f"Используем кешированные результаты для: {query}")
            return self.cache[cache_key]
        
        # Ограничение частоты запросов
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            logger.debug(f"Ожидание {sleep_time:.2f} сек для соблюдения лимита запросов")
            time.sleep(sleep_time)
        
        try:
            with tracer.trace("web_search") as span:
                span.input = query
                
                # Параметры запроса
                params = {
                    "q": query,
                    "format": "json",
                    "no_html": 1,
                    "no_redirect": 1,
                    "api_key": self.api_key,
                    "max_results": self.max_results
                }
                
                # Выполнение запроса
                response = requests.get(
                    "https://api.duckduckgo.com/",
                    params=params,
                    timeout=self.timeout
                )
                
                # Проверка статуса
                if response.status_code != 200:
                    raise Exception(f"API вернуло статус {response.status_code}")
                
                # Парсинг результатов
                results = response.json().get("Results", [])
                logger.info(f"Найдено {len(results)} результатов для '{query}'")
                
                # Обработка результатов
                processed_results = []
                for result in results:
                    processed = self._process_result(result)
                    if processed:
                        processed_results.append(processed)
                
                # Фильтрация по доменам
                filtered_results = filter_domains(
                    processed_results,
                    self.allowed_domains
                )
                
                # Ограничение количества результатов
                final_results = filtered_results[:self.max_results]
                
                # Кеширование
                self.cache[cache_key] = final_results
                self.last_request_time = time.time()
                
                span.output = json.dumps(final_results, ensure_ascii=False)
                return final_results
        
        except Exception as e:
            logger.error(f"Ошибка веб-поиска: {str(e)}")
            tracer.log_error({
                "component": "web_search",
                "query": query,
                "error": str(e)
            })
            return []

    def _process_result(self, result: dict) -> dict:
        """Обрабатывает и очищает результат поиска"""
        try:
            # Базовые поля
            processed = {
                "title": clean_html_tags(result.get("Text", "")),
                "url": result.get("FirstURL", ""),
                "snippet": clean_html_tags(result.get("Result", "")),
                "domain": self._extract_domain(result.get("FirstURL", ""))
            }
            
            # Извлечение иконки
            if "Icon" in result and "URL" in result["Icon"]:
                processed["icon"] = result["Icon"]["URL"]
            
            return processed
        except Exception as e:
            logger.warning(f"Ошибка обработки результата: {str(e)}")
            return {}

    def _extract_domain(self, url: str) -> str:
        """Извлекает домен из URL"""
        if not url:
            return ""
        
        # Упрощенное извлечение домена
        domain = re.sub(r"^https?://(?:www\.)?([^/]+).*$", r"\1", url)
        return domain.lower()

# Утилиты в том же файле для простоты

def clean_html_tags(text: str) -> str:
    """Удаляет HTML-теги из текста"""
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text)

def filter_domains(results: List[dict], allowed_domains: List[str]) -> List[dict]:
    """
    Фильтрует результаты по списку разрешенных доменов
    
    :param results: Список результатов поиска
    :param allowed_domains: Список разрешенных доменов (например, ['example.com', 'docs.ru'])
    :return: Отфильтрованный список результатов
    """
    if not allowed_domains:
        return results
    
    filtered = []
    for result in results:
        domain = result.get("domain", "")
        if any(allowed in domain for allowed in allowed_domains):
            filtered.append(result)
    
    logger.debug(f"Фильтрация: {len(results)} -> {len(filtered)} после применения доменных фильтров")
    return filtered