from langchain_core.runnables import RunnableLambda
from agents import (
    RouterAgent,
    SchemaMasterAgent,
    SQLGeneratorAgent,
    DataInterpreterAgent,
    DataStorytellerAgent,
    QueryClassifierAgent,
    SmallTalkAgent,
    ResponseSynthesizerAgent,
    FallbackAgent,
    GeneralAssistantAgent
)
from functions import (
    sql_validator,
    access_check,
    db_executor,
    rag_search,
    web_search,
    response_formatter
)
from services import MetadataExtractor
from .state import AgentState
import pandas as pd
import logging
import re

# Настройка логгера
logger = logging.getLogger(__name__)

# --------------------------
# Узлы маршрутизации
# --------------------------

def router_node(state: AgentState) -> AgentState:
    """Определяет тип запроса: аналитика, документация или общие вопросы"""
    try:
        router = RouterAgent()
        route = router.route_request(state["user_query"])
        logger.info(f"Routing query to: {route}")
        return {**state, "route": route}
    
    except Exception as e:
        logger.error(f"Router error: {str(e)}")
        return {**state, "error": {"type": "routing", "message": str(e)}}

# --------------------------
# Узлы аналитического пайплайна
# --------------------------

def schema_master_node(state: AgentState) -> AgentState:
    """Получает контекст схемы БД для SQL генерации"""
    if state.get("route") != "analytics":
        return state
    
    try:
        schema_master = SchemaMasterAgent()
        schema_context = schema_master.get_schema_context(state["user_query"])
        logger.debug(f"Schema context retrieved: {schema_context[:100]}...")
        return {**state, "schema_context": schema_context}
    
    except Exception as e:
        logger.error(f"SchemaMaster error: {str(e)}")
        return {
            **state, 
            "error": {
                "type": "schema_retrieval", 
                "message": str(e),
                "query": state["user_query"]
            }
        }

def sql_generator_node(state: AgentState) -> AgentState:
    """Генерирует SQL запрос на основе естественного языка"""
    if not state.get("schema_context") or state.get("route") != "analytics":
        return state
    
    try:
        sql_gen = SQLGeneratorAgent()
        sql_result = sql_gen.generate_sql(
            state["user_query"],
            state["schema_context"]
        )
        logger.info(f"Generated SQL: {sql_result['sql'][:100]}...")
        return {**state, "generated_sql": sql_result["sql"]}
    
    except Exception as e:
        logger.error(f"SQLGenerator error: {str(e)}")
        return {
            **state, 
            "error": {
                "type": "sql_generation", 
                "message": str(e),
                "query": state["user_query"]
            }
        }

def sql_validator_node(state: AgentState) -> AgentState:
    """Проверяет SQL на безопасность и корректность"""
    if not state.get("generated_sql"):
        return state
    
    try:
        if not sql_validator.validate_sql(state["generated_sql"]):
            raise ValueError("SQL validation failed")
        logger.debug("SQL validation passed")
        return state
    
    except Exception as e:
        logger.error(f"SQLValidator error: {str(e)}")
        return {
            **state, 
            "error": {
                "type": "sql_validation", 
                "message": str(e),
                "sql": state["generated_sql"]
            }
        }

def access_check_node(state: AgentState) -> AgentState:
    """Проверяет права доступа пользователя к данным"""
    if not state.get("generated_sql"):
        return state
    
    try:
        if not access_check.check_access(state["user_id"], state["generated_sql"]):
            raise PermissionError("Access denied to requested data")
        logger.debug("Access check passed")
        return state
    
    except Exception as e:
        logger.error(f"AccessCheck error: {str(e)}")
        return {
            **state, 
            "error": {
                "type": "access_denied", 
                "message": str(e),
                "sql": state["generated_sql"]
            }
        }

def db_executor_node(state: AgentState) -> AgentState:
    """Выполняет SQL запрос и возвращает результаты"""
    if state.get("error") or not state.get("generated_sql"):
        return state
    
    try:
        df = db_executor.execute_query(state["generated_sql"])
        logger.info(f"DB query executed, returned {len(df)} rows")
        return {**state, "query_result": df}
    
    except Exception as e:
        logger.error(f"DBExecutor error: {str(e)}")
        return {
            **state, 
            "error": {
                "type": "db_execution", 
                "message": str(e),
                "sql": state["generated_sql"]
            }
        }

def data_interpreter_node(state: AgentState) -> AgentState:
    """Анализирует результаты запроса и извлекает инсайты"""
    if not isinstance(state.get("query_result"), pd.DataFrame):
        return state
    
    try:
        interpreter = DataInterpreterAgent()
        analysis = interpreter.analyze(
            state["query_result"],
            state["user_query"]
        )
        logger.debug(f"Data analysis completed: {analysis['key_insight'][:50]}...")
        return {**state, "analysis_result": analysis}
    
    except Exception as e:
        logger.error(f"DataInterpreter error: {str(e)}")
        return {
            **state, 
            "error": {
                "type": "data_interpretation", 
                "message": str(e),
                "data_sample": state["query_result"].head().to_dict()
            }
        }

# --------------------------
# Узлы документационного пайплайна
# --------------------------

def query_classifier_node(state: AgentState) -> AgentState:
    """Классифицирует документационные запросы"""
    if state.get("route") != "documentation":
        return state
    
    try:
        classifier = QueryClassifierAgent()
        query_type = classifier.classify(state["user_query"])
        logger.info(f"Documentation query classified as: {query_type}")
        return {**state, "doc_query_type": query_type}
    
    except Exception as e:
        logger.error(f"QueryClassifier error: {str(e)}")
        return {
            **state, 
            "error": {
                "type": "query_classification", 
                "message": str(e)
            }
        }

def small_talk_node(state: AgentState) -> AgentState:
    """Обрабатывает разговорные запросы"""
    if state.get("doc_query_type") != "small_talk":
        return state
    
    try:
        agent = SmallTalkAgent()
        response = agent.respond(state["user_query"])
        logger.debug(f"Small talk response: {response[:50]}...")
        return {
            **state, 
            "response_content": response,
            "response_type": "text"
        }
    
    except Exception as e:
        logger.error(f"SmallTalkAgent error: {str(e)}")
        return {
            **state, 
            "error": {
                "type": "small_talk", 
                "message": str(e)
            }
        }

def rag_search_node(state: AgentState) -> AgentState:
    """Ищет информацию в документации через RAG"""
    if state.get("doc_query_type") != "rag_search":
        return state
    
    try:
        results = rag_search.search(state["user_query"])
        logger.info(f"RAG search found {len(results['sources'])} sources")
        return {**state, "rag_results": results}
    
    except Exception as e:
        logger.error(f"RAGSearch error: {str(e)}")
        return {
            **state, 
            "error": {
                "type": "rag_search", 
                "message": str(e)
            }
        }

def web_search_node(state: AgentState) -> AgentState:
    """Ищет актуальную информацию в интернете"""
    if state.get("doc_query_type") != "web_search":
        return state
    
    try:
        results = web_search.search(state["user_query"])
        logger.info(f"Web search found {len(results)} results")
        return {**state, "web_results": results}
    
    except Exception as e:
        logger.error(f"WebSearch error: {str(e)}")
        return {
            **state, 
            "error": {
                "type": "web_search", 
                "message": str(e)
            }
        }

def response_synthesizer_node(state: AgentState) -> AgentState:
    """Синтезирует ответ из нескольких источников"""
    if state.get("route") != "documentation":
        return state
    
    try:
        synthesizer = ResponseSynthesizerAgent()
        
        # Для RAG результатов
        if state.get("rag_results"):
            response = synthesizer.synthesize(
                [state["rag_results"]],
                state["user_query"]
            )
            return {
                **state, 
                "response_content": response,
                "response_type": "text"
            }
        
        # Для веб-результатов
        elif state.get("web_results"):
            content = "\n\n".join([
                f"• {res['title']}: {res['url']}\n{res['snippet']}" 
                for res in state["web_results"]
            ])
            return {
                **state, 
                "response_content": content,
                "response_type": "text"
            }
        
        return state
    
    except Exception as e:
        logger.error(f"ResponseSynthesizer error: {str(e)}")
        return {
            **state, 
            "error": {
                "type": "response_synthesis", 
                "message": str(e)
            }
        }

# --------------------------
# Узлы общего назначения
# --------------------------

def general_assistant_node(state: AgentState) -> AgentState:
    """Обрабатывает общие вопросы"""
    if state.get("route") != "general":
        return state
    
    try:
        assistant = GeneralAssistantAgent()
        response = assistant.respond(state["user_query"])
        logger.debug(f"General assistant response: {response[:50]}...")
        return {
            **state, 
            "response_content": response,
            "response_type": "text"
        }
    
    except Exception as e:
        logger.error(f"GeneralAssistant error: {str(e)}")
        return {
            **state, 
            "error": {
                "type": "general_assistant", 
                "message": str(e)
            }
        }

def fallback_handler_node(state: AgentState) -> AgentState:
    """Обрабатывает ошибки и определяет стратегию восстановления"""
    if not state.get("error"):
        return state
    
    try:
        fallback = FallbackAgent()
        resolution = fallback.handle_error(state["error"])
        logger.warning(
            f"Fallback handling error: {state['error']['type']} -> "
            f"Action: {resolution['action']}"
        )
        
        # Обновляем состояние в зависимости от действия
        new_state = {
            **state,
            "fallback_action": resolution["action"],
            "error": None  # Сбрасываем ошибку
        }
        
        # Дополнительные действия в зависимости от стратегии
        if resolution["action"] == "retry":
            # Сбрасываем состояние для повтора
            return {
                **new_state,
                "generated_sql": None,
                "query_result": None,
                "analysis_result": None
            }
        
        elif resolution["action"] == "simplify":
            # Упрощаем запрос
            return {
                **new_state,
                "route": "general",
                "user_query": resolution.get(
                    "simplified_query", 
                    state["user_query"]
                )
            }
        
        elif resolution["action"] == "notify":
            # Возвращаем сообщение об ошибке
            return {
                **new_state,
                "response_content": resolution["message"],
                "response_type": "text"
            }
        
        return new_state
    
    except Exception as e:
        logger.critical(f"FallbackHandler failed: {str(e)}")
        return {
            **state,
            "response_content": "Критическая ошибка системы. Пожалуйста, попробуйте позже.",
            "response_type": "text"
        }

def data_storyteller_node(state: AgentState) -> AgentState:
    """Выбирает формат представления данных (график, таблица, текст)"""
    if not state.get("analysis_result"):
        return state
    
    try:
        storyteller = DataStorytellerAgent()
        response = storyteller.create_response(
            state["analysis_result"],
            state["query_result"]
        )
        logger.info(f"Response format: {response['type']}")
        return {
            **state, 
            "response_content": response["content"],
            "response_type": response["type"]
        }
    
    except Exception as e:
        logger.error(f"DataStoryteller error: {str(e)}")
        return {
            **state, 
            "error": {
                "type": "data_storytelling", 
                "message": str(e)
            }
        }

def response_formatter_node(state: AgentState) -> AgentState:
    """Форматирует финальный ответ для отправки пользователю"""
    if not state.get("response_content") or not state.get("response_type"):
        # Если нет ответа, но есть результаты аналитики
        if state.get("query_result") and state.get("analysis_result"):
            # Создаем простой текстовый ответ как фолбэк
            return {
                **state,
                "response_content": "Анализ завершен. Используйте /report для получения результатов.",
                "response_type": "text"
            }
        return state
    
    try:
        # Форматируем ответ
        formatted = response_formatter.format({
            "type": state["response_type"],
            "content": state["response_content"]
        })
        
        logger.info(f"Response formatted for {state['response_type']}")
        return {
            **state, 
            "final_response": formatted
        }
    
    except Exception as e:
        logger.error(f"ResponseFormatter error: {str(e)}")
        # Фолбэк: простой текстовый ответ
        return {
            **state,
            "final_response": {
                "type": "text",
                "response": state["response_content"][:1000]  # Обрезаем длинный текст
            }
        }

def metadata_update_node(state: AgentState) -> AgentState:
    """Периодически обновляет метаданные схемы БД (фоновый процесс)"""
    # Этот узел вызывается периодически, не для каждого запроса
    try:
        if not state.get("last_metadata_update") or (pd.Timestamp.now() - state["last_metadata_update"]).days > 1:
            extractor = MetadataExtractor()
            schema = extractor.extract_full_schema()
            extractor.generate_schema_descriptions(schema)
            
            # Сохраняем в векторное хранилище
            # (предполагается, что vector_store доступен через контекст)
            state["vector_store"].index_documents(schema)
            
            logger.info("Database metadata updated")
            return {
                **state, 
                "last_metadata_update": pd.Timestamp.now()
            }
        return state
    
    except Exception as e:
        logger.error(f"Metadata update failed: {str(e)}")
        return state

# --------------------------
# Вспомогательные функции
# --------------------------

def log_state_change(prev_state: dict, new_state: dict):
    """Логирует изменения состояния между узлами"""
    changes = {}
    for key in new_state:
        if key not in prev_state or new_state[key] != prev_state[key]:
            # Сокращаем большие значения для логов
            value = new_state[key]
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            elif isinstance(value, pd.DataFrame):
                value = f"DataFrame({len(value)} rows)"
            
            changes[key] = value
    
    if changes:
        logger.debug(f"State changed: {changes}")

# Обертка для логирования состояния
def logged_node(func):
    def wrapper(state: AgentState) -> AgentState:
        prev_state = state.copy()
        result = func(state)
        log_state_change(prev_state, result)
        return result
    return wrapper

# Применяем логгер ко всем узлам
router_node = logged_node(router_node)
schema_master_node = logged_node(schema_master_node)
sql_generator_node = logged_node(sql_generator_node)
sql_validator_node = logged_node(sql_validator_node)
access_check_node = logged_node(access_check_node)
db_executor_node = logged_node(db_executor_node)
data_interpreter_node = logged_node(data_interpreter_node)
query_classifier_node = logged_node(query_classifier_node)
small_talk_node = logged_node(small_talk_node)
rag_search_node = logged_node(rag_search_node)
web_search_node = logged_node(web_search_node)
response_synthesizer_node = logged_node(response_synthesizer_node)
general_assistant_node = logged_node(general_assistant_node)
fallback_handler_node = logged_node(fallback_handler_node)
data_storyteller_node = logged_node(data_storyteller_node)
response_formatter_node = logged_node(response_formatter_node)
metadata_update_node = logged_node(metadata_update_node)