import requests
import sqlparse
import re
import logging
from typing import List, Dict, Tuple, Optional
from config import Config
from services.monitoring import tracer
import json
import time
from functools import lru_cache

# Настройка логгера
logger = logging.getLogger(__name__)

class AccessChecker:
    def __init__(self):
        # Настройки сервиса аутентификации
        self.auth_service_url = Config.AUTH_SERVICE_URL
        self.auth_token = Config.AUTH_SERVICE_TOKEN
        self.timeout = Config.AUTH_SERVICE_TIMEOUT
        
        # Настройки RBAC
        self.access_matrix = self.load_access_matrix()
        self.max_retries = Config.AUTH_MAX_RETRIES
        self.retry_delay = Config.AUTH_RETRY_DELAY
        
        # Кеш разрешений
        self.permission_cache = {}
        self.cache_ttl = Config.ACCESS_CACHE_TTL

    def check_access(self, user_id: str, sql_query: str) -> Tuple[bool, Dict]:
        """
        Проверяет права доступа пользователя к данным в SQL-запросе
        
        :param user_id: Идентификатор пользователя
        :param sql_query: SQL-запрос для проверки
        :return: Кортеж (разрешен ли доступ, детали проверки)
        """
        result = {
            "has_access": False,
            "user_id": user_id,
            "tables": [],
            "required_roles": {},
            "user_roles": [],
            "errors": []
        }
        
        try:
            with tracer.trace("access_check") as span:
                span.input = json.dumps({"user_id": user_id, "sql_query": sql_query[:500]})
                
                # Шаг 1: Извлечение таблиц из SQL-запроса
                tables = self.extract_tables(sql_query)
                result["tables"] = tables
                
                if not tables:
                    result["errors"].append("Не удалось определить таблицы в запросе")
                    span.output = json.dumps(result)
                    return False, result
                
                # Шаг 2: Получение ролей пользователя
                user_roles = self.get_user_roles(user_id)
                result["user_roles"] = user_roles
                
                if not user_roles:
                    result["errors"].append("Пользователь не имеет назначенных ролей")
                    span.output = json.dumps(result)
                    return False, result
                
                # Шаг 3: Проверка доступа для каждой таблицы
                required_roles = {}
                has_access = True
                
                for table in tables:
                    table_roles = self.get_required_roles(table)
                    required_roles[table] = table_roles
                    
                    # Проверяем, есть ли у пользователя хотя бы одна из требуемых ролей
                    if not any(role in user_roles for role in table_roles):
                        result["errors"].append(f"Доступ к таблице '{table}' запрещен")
                        has_access = False
                
                result["required_roles"] = required_roles
                result["has_access"] = has_access
                
                span.output = json.dumps(result)
                return has_access, result
        
        except Exception as e:
            logger.error(f"Ошибка проверки доступа: {str(e)}")
            result["errors"].append(f"Системная ошибка: {str(e)}")
            return False, result

    def extract_tables(self, sql: str) -> List[str]:
        """Извлекает список таблиц из SQL-запроса"""
        try:
            # Используем sqlparse для точного парсинга
            parsed = sqlparse.parse(sql)
            tables = set()
            
            if not parsed:
                return []
            
            # Функция для рекурсивного поиска таблиц
            def find_tables(token):
                if isinstance(token, sqlparse.sql.Identifier):
                    # Получаем реальное имя таблицы (без алиасов)
                    tables.add(token.get_real_name().lower())
                elif isinstance(token, sqlparse.sql.IdentifierList):
                    for ident in token.get_identifiers():
                        tables.add(ident.get_real_name().lower())
                elif hasattr(token, 'tokens'):
                    for t in token.tokens:
                        find_tables(t)
            
            # Ищем в каждом выражении
            for statement in parsed:
                # Ищем секции FROM и JOIN
                from_seen = False
                for token in statement.tokens:
                    if token.ttype is sqlparse.tokens.Keyword and token.value.lower() == 'from':
                        from_seen = True
                    elif from_seen:
                        if token.ttype is sqlparse.tokens.Keyword and token.value.lower().startswith('join'):
                            # Пропускаем ключевое слово JOIN
                            continue
                        find_tables(token)
            
            return list(tables)
        
        except Exception as e:
            logger.warning(f"Ошибка извлечения таблиц: {str(e)}")
            # Фолбэк: простая регулярка
            tables = re.findall(r'\b(?:from|join)\s+([\w\.]+)', sql, re.IGNORECASE)
            return list(set(table.lower().split('.')[-1] for table in tables))

    @lru_cache(maxsize=1024)
    def get_user_roles(self, user_id: str) -> List[str]:
        """Получает роли пользователя из сервиса аутентификации"""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    f"{self.auth_service_url}/users/{user_id}/roles",
                    headers={"Authorization": f"Bearer {self.auth_token}"},
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json().get("roles", [])
                
                # Обработка ошибок HTTP
                if response.status_code == 404:
                    logger.warning(f"Пользователь {user_id} не найден")
                    return []
                
                if response.status_code == 401:
                    logger.error("Ошибка аутентификации в сервисе прав доступа")
                    return []
                
                logger.warning(f"Сервис прав доступа вернул статус {response.status_code}")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Ошибка подключения к сервису прав доступа: {str(e)}")
            
            # Повторная попытка после задержки
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)
        
        return []

    def get_required_roles(self, table: str) -> List[str]:
        """Возвращает роли, необходимые для доступа к таблице"""
        # Проверка кеша
        if table in self.permission_cache:
            return self.permission_cache[table]
        
        # Поиск в матрице доступа
        table_lower = table.lower()
        roles = []
        
        # Ищем точное совпадение таблицы
        if table_lower in self.access_matrix:
            roles = self.access_matrix[table_lower]
        else:
            # Ищем по шаблонам (например, "sales_*")
            for pattern, pattern_roles in self.access_matrix.items():
                if '*' in pattern:
                    regex = pattern.replace('*', '.*')
                    if re.match(regex, table_lower):
                        roles = pattern_roles
                        break
        
        # Если не найдено, используем политику по умолчанию
        if not roles:
            roles = self.access_matrix.get('default', ['admin'])
        
        # Кешируем результат
        self.permission_cache[table] = roles
        return roles

    def load_access_matrix(self) -> Dict[str, List[str]]:
        """Загружает матрицу доступа из конфигурации или внешнего источника"""
        # В реальной системе это может загружаться из БД или файла конфигурации
        return {
            # Таблицы аналитики
            "sales": ["sales_manager", "analyst", "admin"],
            "customers": ["sales_manager", "customer_support", "admin"],
            "products": ["product_manager", "analyst", "admin"],
            
            # Системные таблицы
            "users": ["admin", "system_admin"],
            "audit_log": ["admin", "auditor"],
            
            # Шаблоны
            "report_*": ["analyst", "report_user"],
            "temp_*": ["analyst", "developer"],
            
            # Политика по умолчанию
            "default": ["admin"]
        }

    def clear_cache(self):
        """Очищает кеш разрешений"""
        self.permission_cache = {}
        self.get_user_roles.cache_clear()

# Пример использования
if __name__ == "__main__":
    # Тестовая конфигурация
    class TestConfig:
        AUTH_SERVICE_URL = "https://auth.example.com/api"
        AUTH_SERVICE_TOKEN = "test_token"
        AUTH_SERVICE_TIMEOUT = 5
        AUTH_MAX_RETRIES = 2
        AUTH_RETRY_DELAY = 1
        ACCESS_CACHE_TTL = 300  # 5 минут
    
    Config = TestConfig()
    
    # Настройка логгирования
    logging.basicConfig(level=logging.INFO)
    
    # Тестовые данные
    user_id = "user_123"
    sql_query = """
    SELECT c.name, SUM(s.amount) AS total
    FROM customers c
    JOIN sales s ON s.customer_id = c.id
    WHERE s.date >= '2023-01-01'
    GROUP BY c.name
    ORDER BY total DESC
    """
    
    # Создание экземпляра и проверка
    checker = AccessChecker()
    
    # 1. Проверка доступа
    has_access, result = checker.check_access(user_id, sql_query)
    print(f"Доступ разрешен: {has_access}")
    print("Детали проверки:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 2. Проверка извлечения таблиц
    tables = checker.extract_tables(sql_query)
    print(f"\nИзвлеченные таблицы: {tables}")
    
    # 3. Получение ролей для таблицы
    print(f"\nРоли для таблицы 'sales': {checker.get_required_roles('sales')}")
    print(f"Роли для таблицы 'users': {checker.get_required_roles('users')}")
    print(f"Роли для таблицы 'report_monthly': {checker.get_required_roles('report_monthly')}")
    print(f"Роли для неизвестной таблицы: {checker.get_required_roles('unknown_table')}")