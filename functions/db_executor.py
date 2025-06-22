import sqlalchemy
from sqlalchemy import text, exc
from sqlalchemy.orm import sessionmaker
import pandas as pd
import logging
import time
import hashlib
import json
from typing import Tuple, Optional, Union
from config import Config
from services.monitoring import tracer
from services.cache_service import CacheService
import re
import numpy as np
import traceback

# Настройка логгера
logger = logging.getLogger(__name__)

class DBExecutor:
    def __init__(self):
        # Инициализация подключения к БД
        self.engine = self.create_engine()
        self.Session = sessionmaker(bind=self.engine)
        self.cache = CacheService()
        self.max_retries = Config.DB_MAX_RETRIES
        self.query_timeout = Config.DB_QUERY_TIMEOUT
        self.max_rows = Config.DB_MAX_ROWS
        self.read_only = Config.DB_READ_ONLY
        self.allow_caching = Config.DB_ALLOW_CACHING
        self.statement_timeout = Config.DB_STATEMENT_TIMEOUT

    def create_engine(self):
        """Создает движок SQLAlchemy с настройками"""
        pool_size = Config.DB_POOL_SIZE or 5
        max_overflow = Config.DB_MAX_OVERFLOW or 10
        pool_recycle = Config.DB_POOL_RECYCLE or 300  # 5 минут
        
        return sqlalchemy.create_engine(
            Config.DB_URL,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_recycle=pool_recycle,
            connect_args={
                "connect_timeout": Config.DB_CONNECT_TIMEOUT or 10
            }
        )

    def execute_query(self, sql: str) -> Tuple[bool, Union[pd.DataFrame, str], dict]:
        """
        Выполняет SQL-запрос и возвращает результат
        
        :param sql: SQL-запрос для выполнения
        :return: Кортеж (успех, результат, метаданные)
        """
        metadata = {
            "sql": sql,
            "cache_key": "",
            "cache_hit": False,
            "execution_time": 0,
            "row_count": 0,
            "error": None
        }
        
        try:
            # Генерация ключа кеша
            cache_key = self.generate_cache_key(sql)
            metadata["cache_key"] = cache_key
            
            # Проверка кеша
            if self.allow_caching:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    logger.info(f"Кеш-попадание для запроса: {cache_key[:20]}...")
                    metadata["cache_hit"] = True
                    return True, pd.read_json(cached_result, orient='split'), metadata
            
            # Проверка на read-only режим
            if self.read_only and not self.is_read_only_query(sql):
                error_msg = "Попытка выполнения запроса на запись в read-only режиме"
                logger.warning(error_msg)
                metadata["error"] = error_msg
                return False, error_msg, metadata
            
            start_time = time.time()
            
            # Выполнение запроса с повтором при ошибках
            for attempt in range(self.max_retries):
                try:
                    with self.engine.connect().execution_options(
                        isolation_level="AUTOCOMMIT",
                        stream_results=True
                    ) as conn:
                        # Установка таймаута
                        if self.statement_timeout:
                            conn.execute(text(f"SET statement_timeout = {self.statement_timeout * 1000}"))
                        
                        # Выполнение запроса
                        result = conn.execute(
                            text(sql).execution_options(
                                timeout=self.query_timeout
                            )
                        )
                        
                        # Получение результатов с ограничением по количеству строк
                        rows = []
                        for i, row in enumerate(result):
                            if i >= self.max_rows:
                                logger.warning(f"Превышено максимальное количество строк ({self.max_rows}), обрезание результатов")
                                break
                            rows.append(dict(row))
                        
                        # Создание DataFrame
                        if rows:
                            df = pd.DataFrame(rows)
                        else:
                            # Для пустых результатов
                            df = pd.DataFrame(columns=[col.name for col in result.cursor.description])
                        
                        metadata["row_count"] = len(df)
                        metadata["execution_time"] = time.time() - start_time
                        
                        # Кеширование результата
                        if self.allow_caching:
                            self.cache.set(
                                cache_key, 
                                df.to_json(orient='split'),
                                ex=Config.DB_CACHE_TTL
                            )
                        
                        return True, df, metadata
                
                except exc.OperationalError as e:
                    # Обработка временных ошибок (например, потеря соединения)
                    if "connection" in str(e).lower() and attempt < self.max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        logger.warning(f"Ошибка подключения (попытка {attempt+1}): {str(e)}. Повтор через {wait_time} сек.")
                        time.sleep(wait_time)
                        continue
                    raise
                
                except exc.StatementError as e:
                    # Ошибки выполнения запроса
                    if "timeout" in str(e).lower() and attempt < self.max_retries - 1:
                        logger.warning(f"Таймаут запроса (попытка {attempt+1}). Повтор через 1 сек.")
                        time.sleep(1)
                        continue
                    raise
        
        except Exception as e:
            error_msg = self.format_db_error(e)
            logger.error(f"Ошибка выполнения запроса: {error_msg}")
            metadata["error"] = error_msg
            metadata["execution_time"] = time.time() - start_time
            return False, error_msg, metadata

    def generate_cache_key(self, sql: str) -> str:
        """Генерирует уникальный ключ кеша для SQL-запроса"""
        # Нормализация SQL для устранения незначительных различий
        normalized_sql = re.sub(r'\s+', ' ', sql).strip().lower()
        normalized_sql = re.sub(r'[\'"](.*?)[\'"]', '?', normalized_sql)  # Удаляем литералы
        return f"sql:{hashlib.sha256(normalized_sql.encode()).hexdigest()}"

    def is_read_only_query(self, sql: str) -> bool:
        """Проверяет, является ли запрос только для чтения"""
        # Используем sqlparse для анализа
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return False
                
            first_token = parsed[0].token_first()
            return first_token and first_token.ttype is sqlparse.tokens.DML and first_token.value.lower() == "select"
        except Exception:
            # Фолбэк: простая проверка
            return sql.strip().lower().startswith("select")

    def format_db_error(self, error: Exception) -> str:
        """Форматирует ошибку БД для пользователя"""
        # Определяем тип ошибки
        if isinstance(error, exc.ProgrammingError):
            return "Синтаксическая ошибка SQL"
        elif isinstance(error, exc.OperationalError):
            return "Проблема с подключением к базе данных"
        elif isinstance(error, exc.IntegrityError):
            return "Ошибка целостности данных (например, нарушение уникальности)"
        elif isinstance(error, exc.DataError):
            return "Ошибка данных (некорректные значения)"
        elif "timeout" in str(error).lower():
            return "Таймаут выполнения запроса"
        elif "connection" in str(error).lower():
            return "Ошибка подключения к базе данных"
        
        # Общая ошибка
        return "Ошибка при выполнении запроса к базе данных"

    def get_sample_data(self, table: str, limit: int = 5) -> Tuple[bool, Union[pd.DataFrame, str], dict]:
        """Возвращает образец данных из таблицы"""
        sql = f"SELECT * FROM {table} LIMIT {limit}"
        return self.execute_query(sql)

    def test_connection(self) -> bool:
        """Проверяет подключение к базе данных"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Ошибка подключения к БД: {str(e)}")
            return False