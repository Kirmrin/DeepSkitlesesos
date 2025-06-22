import os

class Config:
    # Настройки Telegram
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_WEBHOOK_URL = os.getenv("TELEGRAM_WEBHOOK_URL", "https://your-domain.com/webhook")
    
    # Настройки LLM (GigaChat)
    GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY")
    GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")
    
    # Базы данных
    DB_URL = os.getenv("DB_URL", "postgresql://user:pass@localhost/dbname")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Векторное хранилище
    VECTOR_STORE_URL = os.getenv("VECTOR_STORE_URL")
    VECTOR_STORE_API_KEY = os.getenv("VECTOR_STORE_API_KEY")
    
    # Внешние API
    JIRA_API_URL = os.getenv("JIRA_API_URL")
    JIRA_USER = os.getenv("JIRA_USER")
    JIRA_TOKEN = os.getenv("JIRA_TOKEN")
    SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK")
    DUCKDUCKGO_API_KEY = os.getenv("DUCKDUCKGO_API_KEY")
    
    # LangSmith
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "telegram-analytics-system")
    
    # Настройки системы
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
    CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))  # 1 час
    SQL_EXECUTION_TIMEOUT = int(os.getenv("SQL_EXECUTION_TIMEOUT", 30))