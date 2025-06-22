from typing import TypedDict, Optional, Union, Any, Dict, List
import pandas as pd
from pydantic import BaseModel

class AgentState(TypedDict):
    # Входные данные
    user_id: str
    chat_id: int
    user_query: str
    
    # Маршрутизация
    route: Optional[str]  # 'analytics', 'documentation', 'general'
    
    # Аналитика
    schema_context: Optional[str]
    generated_sql: Optional[str]
    query_result: Optional[Union[pd.DataFrame, List[Dict]]]
    analysis_result: Optional[Dict]
    
    # Документация
    doc_query_type: Optional[str]  # 'small_talk', 'rag_search', 'web_search'
    rag_results: Optional[Dict]
    web_results: Optional[List[Dict]]
    
    # Ошибки
    error: Optional[Dict]
    fallback_action: Optional[str]  # 'retry', 'simplify', 'escalate'
    
    # Выход
    response_content: Optional[str]
    response_type: Optional[str]  # 'text', 'image', 'table'
    final_response: Optional[Dict]  # Форматированный ответ для Telegram