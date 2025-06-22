import logging
import config
import json
import time
import hashlib
import redis
from datetime import timedelta
from typing import Any, Dict, Optional, Callable

class CacheService:
    def __init__(self):
        self.logger = logging.getLogger("cache_service")
        self.memory_cache = {}
        self.redis_client = self._init_redis()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "memory_size": 0,
            "redis_size": 0
        }
        
    def _init_redis(self) -> Optional[redis.Redis]:
        """Инициализация подключения к Redis"""
        if not config.REDIS_ENABLED:
            return None
            
        try:
            return redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                password=config.REDIS_PASSWORD,
                db=config.REDIS_DB,
                decode_responses=True
            )
        except Exception as e:
            self.logger.error(f"Redis connection failed: {str(e)}")
            return None

    def get(self, key: str, loader: Callable = None, ttl: int = None) -> Any:
        """
        Получает значение из кеша. Если нет - загружает через loader.
        :param key: Ключ кеша
        :param loader: Функция для загрузки данных при отсутствии в кеше
        :param ttl: Время жизни записи в секундах
        :return: Значение из кеша или результат loader
        """
        # 1. Попытка получить из памяти
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() < entry["expire"]:
                self.stats["hits"] += 1
                return entry["value"]
            del self.memory_cache[key]
            self.stats["memory_size"] -= 1

        # 2. Попытка получить из Redis
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    self.stats["hits"] += 1
                    # Сохраняем в memory для ускорения
                    self._set_memory(key, value, ttl or config.DEFAULT_MEMORY_TTL)
                    return json.loads(value)
            except redis.RedisError as e:
                self.logger.warning(f"Redis get failed: {str(e)}")

        # 3. Загрузка через функцию
        self.stats["misses"] += 1
        if loader:
            value = loader()
            self.set(key, value, ttl)
            return value
            
        return None

    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """
        Сохраняет значение в кеш
        :param key: Ключ кеша
        :param value: Значение (любой JSON-сериализуемый объект)
        :param ttl: Время жизни в секундах
        """
        # Определение TTL
        actual_ttl = ttl or config.DEFAULT_CACHE_TTL
        
        # Сохранение в memory
        self._set_memory(key, value, actual_ttl)
        
        # Сохранение в Redis
        if self.redis_client:
            try:
                serialized = json.dumps(value)
                self.redis_client.setex(key, actual_ttl, serialized)
                self.stats["redis_size"] = self.redis_client.dbsize()
            except redis.RedisError as e:
                self.logger.warning(f"Redis set failed: {str(e)}")

    def _set_memory(self, key: str, value: Any, ttl: int) -> None:
        """Сохраняет значение в памяти с учетом TTL"""
        # Проверка лимита памяти
        if len(self.memory_cache) >= config.MAX_MEMORY_ENTRIES:
            self._evict_old_memory_entries()
            
        self.memory_cache[key] = {
            "value": value,
            "expire": time.time() + ttl,
            "created": time.time()
        }
        self.stats["memory_size"] = len(self.memory_cache)

    def _evict_old_memory_entries(self, num_to_evict: int = 5) -> None:
        """Удаляет старые записи из memory-кеша"""
        # Сортировка по времени создания
        sorted_keys = sorted(
            self.memory_cache.keys(),
            key=lambda k: self.memory_cache[k]["created"]
        )[:num_to_evict]
        
        for key in sorted_keys:
            del self.memory_cache[key]
            
        self.stats["memory_size"] = len(self.memory_cache)

    def delete(self, key: str) -> None:
        """Удаляет значение из всех уровней кеша"""
        # Удаление из памяти
        if key in self.memory_cache:
            del self.memory_cache[key]
            self.stats["memory_size"] = len(self.memory_cache)
            
        # Удаление из Redis
        if self.redis_client:
            try:
                self.redis_client.delete(key)
                self.stats["redis_size"] = self.redis_client.dbsize()
            except redis.RedisError as e:
                self.logger.warning(f"Redis delete failed: {str(e)}")

    def clear(self, level: str = "all") -> None:
        """
        Очищает кеш
        :param level: 'memory', 'redis', 'all'
        """
        if level in ("memory", "all"):
            self.memory_cache = {}
            self.stats["memory_size"] = 0
            
        if level in ("redis", "all") and self.redis_client:
            try:
                self.redis_client.flushdb()
                self.stats["redis_size"] = 0
            except redis.RedisError as e:
                self.logger.error(f"Redis flush failed: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику использования кеша"""
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "memory_entries": self.stats["memory_size"],
            "redis_entries": self.stats["redis_size"],
            "hit_rate": self.stats["hits"] / max(1, self.stats["hits"] + self.stats["misses"])
        }

    def generate_key(self, *args, **kwargs) -> str:
        """
        Генерирует уникальный ключ кеша на основе параметров
        :param args: Позиционные аргументы
        :param kwargs: Именованные аргументы
        :return: Хеш-ключ
        """
        parts = [str(arg) for arg in args]
        parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        input_string = "&".join(parts)
        return hashlib.sha256(input_string.encode()).hexdigest()

    def memoize(self, ttl: int = None, key_func: Callable = None):
        """
        Декоратор для кеширования результатов функций
        :param ttl: Время жизни кеша
        :param key_func: Функция для генерации ключа
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Генерация ключа
                if key_func:
                    key = key_func(*args, **kwargs)
                else:
                    key = self.generate_key(func.__name__, *args, **kwargs)
                
                # Попытка получить из кеша
                cached = self.get(key)
                if cached is not None:
                    return cached
                
                # Выполнение функции и сохранение результата
                result = func(*args, **kwargs)
                self.set(key, result, ttl)
                return result
            return wrapper
        return decorator

    def invalidate_by_pattern(self, pattern: str) -> None:
        """
        Инвалидирует записи по шаблону ключа
        :param pattern: Шаблон для поиска ключей (например "user:*")
        """
        # Для memory кеша
        keys_to_delete = [k for k in self.memory_cache if pattern in k]
        for key in keys_to_delete:
            del self.memory_cache[key]
        self.stats["memory_size"] = len(self.memory_cache)
        
        # Для Redis
        if self.redis_client:
            try:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            except redis.RedisError as e:
                self.logger.error(f"Redis pattern delete failed: {str(e)}")