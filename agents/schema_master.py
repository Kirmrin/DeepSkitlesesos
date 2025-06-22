from langsmith import traceable
from gigachain import GigaChatModel
import logging
import config
import re
import json
from typing import Dict, List, Optional, Tuple

class SchemaMaster:
    def __init__(self, db_schema: Dict):
        self.logger = logging.getLogger("schema_master")
        self.llm = GigaChatModel(
            model="GigaChat-Pro",
            temperature=0.1,  # Минимум случайности для точности
            api_key=config.GIGACHAT_API_KEY,
            max_tokens=500
        )
        self.schema_version = "1.2"
        self.schema = self._validate_schema(db_schema)
        self.schema_cache = {}
        self.query_cache = {}
        
    def _validate_schema(self, schema: Dict) -> Dict:
        """Проверяет и нормализует схему БД"""
        required_keys = ["tables", "relationships", "metadata"]
        if not all(key in schema for key in required_keys):
            raise ValueError("Invalid schema format")
            
        # Нормализация названий таблиц/колонок
        for table in schema["tables"]:
            table["name"] = table["name"].lower()
            for column in table["columns"]:
                column["name"] = column["name"].lower()
                
        return schema

    @traceable
    def generate_sql(self, nl_query: str, user_context: Dict) -> Dict:
        """
        Генерирует SQL-запрос из естественно-языкового описания
        
        :param nl_query: Запрос на естественном языке ("Покажи топ-10 клиентов по продажам")
        :param user_context: Контекст пользователя (роль, доступ)
        :return: {
            "sql": "SELECT ...", 
            "validated": bool,
            "explanation": str,
            "optimized": bool,
            "security_checked": bool
        }
        """
        # Проверка кеша
        cache_key = f"{nl_query[:50]}-{user_context.get('role','')}"
        if cache_key in self.query_cache:
            self.logger.debug("Using cached SQL query")
            return self.query_cache[cache_key]
        
        try:
            # Генерация SQL через LLM
            raw_sql = self._generate_with_llm(nl_query, user_context)
            
            # Валидация и оптимизация
            result = self._validate_and_optimize(raw_sql, user_context)
            
            # Кеширование
            self.query_cache[cache_key] = result
            return result
            
        except Exception as e:
            self.logger.error(f"SQL generation failed: {str(e)}")
            return {
                "sql": "",
                "validated": False,
                "explanation": f"Ошибка генерации: {str(e)}",
                "optimized": False,
                "security_checked": False
            }

    def _generate_with_llm(self, nl_query: str, context: Dict) -> str:
        """Генерация SQL через LLM с учетом схемы"""
        schema_context = self._get_relevant_schema(nl_query)
        
        prompt = f"""Ты senior SQL разработчик. Сгенерируй PostgreSQL запрос для аналитической системы.

### Схема БД (версия {self.schema_version}):
{schema_context}

### Контекст пользователя:
- Роль: {context.get('role', 'analyst')}
- Доступные таблицы: {', '.join(context.get('allowed_tables', list(self.schema['tables'].keys())))}

### Запрос:
{nl_query}

### Требования:
1. Только SELECT запросы
2. Используй стандартный SQL (без специфичных функций)
3. Всегда указывай таблицы для колонок (table.column)
4. Никаких DML/DDL операций
5. Учитывай ограничения ролей

### Ответ:
{{"sql": "SQL_QUERY", "explanation": "Краткое объяснение"}}
"""
        
        response = self.llm.generate(prompt, response_format="json")
        result = json.loads(response)
        return result["sql"]

    def _get_relevant_schema(self, nl_query: str) -> str:
        """Извлекает релевантные части схемы для запроса"""
        # Определение ключевых сущностей
        entities = self._extract_entities(nl_query)
        
        # Если найдены сущности - возвращаем связанные таблицы
        if entities:
            return self._describe_related_tables(entities)
        
        # Иначе возвращаем всю схему (сжатый формат)
        return self._get_full_schema_summary()

    def _extract_entities(self, text: str) -> List[str]:
        """Извлекает ключевые сущности из запроса"""
        prompt = f"""Извлеки ключевые бизнес-сущности из запроса:
        
Запрос: "{text}"

Выведи JSON-список: ["сущность1", "сущность2"]"""
        
        response = self.llm.generate(prompt, response_format="json")
        return json.loads(response)

    def _describe_related_tables(self, entities: List[str]) -> str:
        """Описание таблиц, связанных с сущностями"""
        schema_description = []
        
        for entity in entities:
            entity_lower = entity.lower()
            # Поиск таблиц по имени или описанию
            for table in self.schema["tables"]:
                if (entity_lower in table["name"] or 
                    entity_lower in table.get("description", "").lower()):
                    
                    # Форматирование описания таблицы
                    desc = f"Таблица: {table['name']}\n"
                    desc += f"Описание: {table.get('description', 'нет описания')}\n"
                    desc += "Колонки:\n"
                    
                    for col in table["columns"]:
                        desc += f"- {col['name']}: {col['type']}"
                        if "description" in col:
                            desc += f" ({col['description']})"
                        desc += "\n"
                    
                    schema_description.append(desc)
        
        return "\n\n".join(schema_description) or self._get_full_schema_summary()

    def _get_full_schema_summary(self) -> str:
        """Возвращает сжатое описание всей схемы"""
        if "full_summary" in self.schema_cache:
            return self.schema_cache["full_summary"]
            
        summary = f"База: {self.schema['metadata']['db_name']}\n"
        summary += f"Таблицы ({len(self.schema['tables'])}):\n"
        
        for table in self.schema["tables"]:
            summary += f"- {table['name']}: {table.get('description', '')} "
            summary += f"({len(table['columns'])} колонок)\n"
        
        self.schema_cache["full_summary"] = summary
        return summary

    def _validate_and_optimize(self, sql: str, context: Dict) -> Dict:
        """Проверка и оптимизация SQL-запроса"""
        # 1. Проверка безопасности
        security_check = self._security_validation(sql, context)
        if not security_check["valid"]:
            return {
                "sql": sql,
                "validated": False,
                "explanation": security_check["reason"],
                "optimized": False,
                "security_checked": False
            }
        
        # 2. Синтаксическая проверка
        syntax_check = self._syntax_validation(sql)
        if not syntax_check["valid"]:
            return {
                "sql": sql,
                "validated": False,
                "explanation": syntax_check["reason"],
                "optimized": False,
                "security_checked": True
            }
        
        # 3. Оптимизация
        optimized_sql, optimization_report = self._optimize_query(sql)
        
        return {
            "sql": optimized_sql,
            "validated": True,
            "explanation": optimization_report,
            "optimized": True,
            "security_checked": True
        }

    def _security_validation(self, sql: str, context: Dict) -> Dict:
        """Проверка на соответствие политикам безопасности"""
        # 1. Проверка запрещенных операций
        forbidden_patterns = [
            r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|GRANT|ALTER)\b",
            r";\s*--",
            r"\b(1=1|TRUE)\b"
        ]
        
        for pattern in forbidden_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                return {
                    "valid": False,
                    "reason": f"Обнаружена запрещенная операция: {pattern}"
                }
        
        # 2. Проверка доступности таблиц
        allowed_tables = context.get("allowed_tables", [])
        if allowed_tables:
            table_pattern = r"\b(FROM|JOIN)\s+(\w+)"
            used_tables = set()
            
            for match in re.finditer(table_pattern, sql, re.IGNORECASE):
                table_name = match.group(2).lower()
                if table_name not in allowed_tables:
                    return {
                        "valid": False,
                        "reason": f"Нет доступа к таблице: {table_name}"
                    }
                used_tables.add(table_name)
        
        # 3. Проверка чувствительных колонок
        sensitive_columns = ["password", "token", "credit_card"]
        for col in sensitive_columns:
            if re.search(rf"\b{col}\b", sql, re.IGNORECASE):
                return {
                    "valid": False,
                    "reason": f"Попытка доступа к чувствительной колонке: {col}"
                }
        
        return {"valid": True, "reason": "Проверка пройдена"}

    def _syntax_validation(self, sql: str) -> Dict:
        """Проверка синтаксиса через LLM"""
        prompt = f"""Проверь SQL запрос на синтаксические ошибки (PostgreSQL):

Запрос:
{sql}

Если есть ошибки - исправь их и объясни. Ответ в формате:
{{
  "valid": true/false,
  "corrected_sql": "ИСПРАВЛЕННЫЙ_SQL",
  "errors": ["ошибка1", "ошибка2"]
}}"""
        
        response = self.llm.generate(prompt, response_format="json")
        result = json.loads(response)
        
        if result["valid"]:
            return {"valid": True, "reason": "Синтаксис корректен"}
            
        return {
            "valid": False,
            "reason": f"Ошибки: {', '.join(result['errors'])}. Исправленный запрос: {result['corrected_sql']}"
        }

    def _optimize_query(self, sql: str) -> Tuple[str, str]:
        """Оптимизация SQL-запроса"""
        prompt = f"""Оптимизируй SQL запрос для аналитической БД (PostgreSQL):

Исходный запрос:
{sql}

Учти:
- Используй современные методы оптимизации
- Добавь комментарии с объяснением изменений
- Сохрани функциональность

Ответ в формате:
{{
  "optimized_sql": "ОПТИМИЗИРОВАННЫЙ_SQL",
  "optimizations": ["список изменений"]
}}"""
        
        response = self.llm.generate(prompt, response_format="json")
        result = json.loads(response)
        
        return result["optimized_sql"], "; ".join(result["optimizations"])

    def get_table_description(self, table_name: str) -> Optional[Dict]:
        """Возвращает описание таблицы по имени"""
        table_name = table_name.lower()
        for table in self.schema["tables"]:
            if table["name"] == table_name:
                return table
        return None

    def get_column_info(self, table_name: str, column_name: str) -> Optional[Dict]:
        """Возвращает информацию о колонке"""
        table = self.get_table_description(table_name)
        if not table:
            return None
            
        column_name = column_name.lower()
        for col in table["columns"]:
            if col["name"] == column_name:
                return col
        return None

    def explain_schema_changes(self, old_schema: Dict, new_schema: Dict) -> str:
        """Генерирует объяснение изменений схемы"""
        prompt = f"""Сравни две версии схемы БД и объясни изменения:

### Старая схема (v{old_schema.get('version', '1.0')}):
{json.dumps(old_schema, indent=2)}

### Новая схема (v{new_schema.get('version', '2.0')}):
{json.dumps(new_schema, indent=2)}

Сформулируй краткий отчет:
- Добавленные таблицы/колонки
- Удаленные элементы
- Измененные структуры
- Влияние на существующие запросы"""
        
        return self.llm.generate(prompt)

    def clear_cache(self):
        """Очищает кеши"""
        self.schema_cache = {}
        self.query_cache = {}
        self.logger.info("SchemaMaster caches cleared")