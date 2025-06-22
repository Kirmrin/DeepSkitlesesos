import sqlparse
from sqlparse.sql import Statement, Token
from sqlparse.tokens import DML, Keyword, Wildcard
from config import Config
from services.monitoring import tracer
import logging
from typing import List, Dict, Tuple, Optional
import re

# Настройка логгера
logger = logging.getLogger(__name__)

class SQLValidator:
    def __init__(self):
        # Запрещенные ключевые слова
        self.forbidden_keywords = [
            "insert", "update", "delete", "drop", "truncate", 
            "alter", "create", "grant", "revoke", "exec", 
            "execute", "shutdown", "backup", "restore"
        ]
        
        # Разрешенные ключевые слова (только для SELECT)
        self.allowed_keywords = [
            "select", "from", "where", "join", "group by", "order by",
            "having", "limit", "offset", "as", "distinct", "case", "when"
        ]
        
        # Максимальная сложность запроса
        self.max_joins = Config.SQL_MAX_JOINS or 5
        self.max_conditions = Config.SQL_MAX_CONDITIONS or 10
        self.max_subqueries = Config.SQL_MAX_SUBQUERIES or 2
        
        # Регулярные выражения для опасных конструкций
        self.dangerous_patterns = [
            r";\s*--",      # SQL-инъекции через комментарий
            r"union\s+all", # Возможные UNION-based атаки
            r"xp_cmdshell", # Опасные процедуры
            r"waitfor\s+delay", # Time-based атаки
            r"dbcc",        # Команды консоли
            r"\.\./\.\./",  # Path traversal
        ]

    def validate_sql(self, sql: str) -> Tuple[bool, Dict]:
        """
        Проводит комплексную валидацию SQL-запроса
        
        :param sql: SQL-запрос для проверки
        :return: Кортеж (валиден ли запрос, словарь с деталями проверки)
        """
        validation_result = {
            "is_valid": False,
            "checks": {
                "forbidden_keywords": False,
                "only_select": False,
                "syntax_check": False,
                "dangerous_patterns": False,
                "complexity_check": False
            },
            "details": {},
            "errors": []
        }
        
        try:
            with tracer.trace("sql_validation") as span:
                span.input = sql
                
                # Проверка 1: Пустой запрос
                if not sql.strip():
                    validation_result["errors"].append("Пустой SQL-запрос")
                    span.output = json.dumps(validation_result)
                    return False, validation_result
                
                # Нормализация SQL
                normalized_sql = self.normalize_sql(sql)
                
                # Проверка 2: Запрещенные ключевые слова
                if not self.check_forbidden_keywords(normalized_sql, validation_result):
                    validation_result["errors"].append("Обнаружены запрещенные ключевые слова")
                
                # Проверка 3: Только SELECT запросы
                if not self.check_only_select(normalized_sql, validation_result):
                    validation_result["errors"].append("Разрешены только SELECT запросы")
                
                # Проверка 4: Опасные паттерны
                if not self.check_dangerous_patterns(normalized_sql, validation_result):
                    validation_result["errors"].append("Обнаружены опасные конструкции")
                
                # Проверка 5: Синтаксический анализ
                parsed, syntax_ok = self.check_syntax(normalized_sql, validation_result)
                if not syntax_ok:
                    validation_result["errors"].append("Синтаксическая ошибка")
                
                # Проверка 6: Сложность запроса (только если синтаксис верен)
                if syntax_ok and parsed:
                    if not self.check_complexity(parsed, validation_result):
                        validation_result["errors"].append("Превышена допустимая сложность запроса")
                
                # Финальная проверка
                is_valid = not validation_result["errors"]
                validation_result["is_valid"] = is_valid
                
                span.output = json.dumps(validation_result)
                return is_valid, validation_result
        
        except Exception as e:
            logger.error(f"Ошибка валидации SQL: {str(e)}")
            validation_result["errors"].append(f"Системная ошибка: {str(e)}")
            return False, validation_result

    def normalize_sql(self, sql: str) -> str:
        """Нормализует SQL для анализа"""
        # Приведение к нижнему регистру
        normalized = sql.lower()
        
        # Удаление комментариев
        normalized = re.sub(r"--.*?$", "", normalized, flags=re.MULTILINE)
        normalized = re.sub(r"/\*.*?\*/", "", normalized, flags=re.DOTALL)
        
        # Замена множественных пробелов
        normalized = re.sub(r"\s+", " ", normalized).strip()
        
        return normalized

    def check_forbidden_keywords(self, sql: str, result: Dict) -> bool:
        """Проверяет наличие запрещенных ключевых слов"""
        found_forbidden = []
        
        for keyword in self.forbidden_keywords:
            # Ищем целые слова, чтобы избежать ложных срабатываний
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, sql):
                found_forbidden.append(keyword)
        
        result["checks"]["forbidden_keywords"] = len(found_forbidden) == 0
        result["details"]["forbidden_keywords_found"] = found_forbidden
        
        return len(found_forbidden) == 0

    def check_only_select(self, sql: str, result: Dict) -> bool:
        """Проверяет, что запрос является SELECT"""
        # Используем sqlparse для точного определения типа
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return False
                
            first_token = parsed[0].token_first()
            is_select = first_token and first_token.ttype is DML and first_token.value.lower() == "select"
            
            result["checks"]["only_select"] = is_select
            return is_select
        except Exception:
            # Фолбэк: простая проверка
            is_select = sql.strip().lower().startswith("select")
            result["checks"]["only_select"] = is_select
            return is_select

    def check_dangerous_patterns(self, sql: str, result: Dict) -> bool:
        """Проверяет опасные паттерны"""
        found_patterns = []
        
        for pattern in self.dangerous_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                found_patterns.append(pattern)
        
        result["checks"]["dangerous_patterns"] = len(found_patterns) == 0
        result["details"]["dangerous_patterns_found"] = found_patterns
        
        return len(found_patterns) == 0

    def check_syntax(self, sql: str, result: Dict) -> Tuple[Optional[List], bool]:
        """Проверяет базовый синтаксис SQL"""
        try:
            parsed = sqlparse.parse(sql)
            is_valid = bool(parsed) and all(isinstance(stmt, Statement) for stmt in parsed)
            
            result["checks"]["syntax_check"] = is_valid
            result["details"]["statements"] = len(parsed) if parsed else 0
            
            return parsed, is_valid
        except Exception as e:
            logger.warning(f"Ошибка синтаксического анализа: {str(e)}")
            result["checks"]["syntax_check"] = False
            result["details"]["syntax_error"] = str(e)
            return None, False

    def check_complexity(self, parsed: List, result: Dict) -> bool:
        """Проверяет сложность запроса"""
        complexity = {
            "tables": 0,
            "joins": 0,
            "conditions": 0,
            "subqueries": 0,
            "functions": 0
        }
        
        # Анализируем каждое выражение
        for stmt in parsed:
            # Считаем таблицы
            from_seen = False
            for token in stmt.tokens:
                if not from_seen:
                    if token.ttype is Keyword and token.value.lower() == "from":
                        from_seen = True
                else:
                    if isinstance(token, sqlparse.sql.Identifier):
                        complexity["tables"] += 1
                    elif token.ttype is Keyword and token.value.lower().startswith("join"):
                        complexity["joins"] += 1
                
                # Считаем условия (WHERE, HAVING)
                if token.ttype is Keyword and token.value.lower() in ("where", "having"):
                    complexity["conditions"] += self.count_conditions(token)
                
                # Считаем подзапросы
                if token.ttype is Keyword and token.value.lower() == "select" and self.is_subquery(token):
                    complexity["subqueries"] += 1
                
                # Считаем функции
                if token.ttype is sqlparse.tokens.Name and token.value.endswith("("):
                    complexity["functions"] += 1
        
        # Проверяем ограничения
        is_valid = True
        if complexity["joins"] > self.max_joins:
            result["errors"].append(f"Слишком много JOIN ({complexity['joins']} > {self.max_joins})")
            is_valid = False
        
        if complexity["conditions"] > self.max_conditions:
            result["errors"].append(f"Слишком много условий ({complexity['conditions']} > {self.max_conditions})")
            is_valid = False
        
        if complexity["subqueries"] > self.max_subqueries:
            result["errors"].append(f"Слишком много подзапросов ({complexity['subqueries']} > {self.max_subqueries})")
            is_valid = False
        
        result["checks"]["complexity_check"] = is_valid
        result["details"]["complexity"] = complexity
        
        return is_valid

    def count_conditions(self, token) -> int:
        """Считает количество условий в WHERE/HAVING"""
        count = 0
        # Простая эвристика: считаем AND/OR
        for t in token.flatten():
            if t.ttype is Keyword and t.value.lower() in ("and", "or"):
                count += 1
        return max(count, 1)  # Минимум 1 условие

    def is_subquery(self, token) -> bool:
        """Определяет, является ли токен подзапросом"""
        # Проверяем, есть ли скобки вокруг
        parent = token.parent
        while parent:
            if parent.ttype is sqlparse.tokens.Punctuation and parent.value == "(":
                return True
            parent = parent.parent
        return False

# Утилиты для работы с SQL

def extract_tables(sql: str) -> List[str]:
    """Извлекает список таблиц из SQL-запроса"""
    try:
        parsed = sqlparse.parse(sql)
        tables = set()
        
        for stmt in parsed:
            for token in stmt.tokens:
                if isinstance(token, sqlparse.sql.Identifier):
                    tables.add(token.get_real_name())
                elif isinstance(token, sqlparse.sql.IdentifierList):
                    for ident in token.get_identifiers():
                        tables.add(ident.get_real_name())
        
        return list(tables)
    except Exception:
        return []