from langgraph.graph import StateGraph, END
from .nodes import (
    router_node,
    schema_master_node,
    sql_generator_node,
    sql_validator_node,
    access_check_node,
    db_executor_node,
    data_interpreter_node,
    query_classifier_node,
    small_talk_node,
    rag_search_node,
    web_search_node,
    response_synthesizer_node,
    general_assistant_node,
    fallback_handler_node,
    data_storyteller_node,
    response_formatter_node
)
from .state import AgentState
from langchain_core.runnables import RunnableLambda
import logging

# Настройка логгера
logger = logging.getLogger(__name__)

def build_telegram_analytics_graph():
    """Создает и конфигурирует граф обработки запросов"""
    # Инициализация графа
    graph = StateGraph(AgentState)
    
    # ------------------------------
    # Регистрация всех узлов графа
    # ------------------------------
    nodes = [
        ("router", RunnableLambda(router_node)),
        ("schema_master", RunnableLambda(schema_master_node)),
        ("sql_generator", RunnableLambda(sql_generator_node)),
        ("sql_validator", RunnableLambda(sql_validator_node)),
        ("access_check", RunnableLambda(access_check_node)),
        ("db_executor", RunnableLambda(db_executor_node)),
        ("data_interpreter", RunnableLambda(data_interpreter_node)),
        ("data_storyteller", RunnableLambda(data_storyteller_node)),
        ("query_classifier", RunnableLambda(query_classifier_node)),
        ("small_talk", RunnableLambda(small_talk_node)),
        ("rag_search", RunnableLambda(rag_search_node)),
        ("web_search", RunnableLambda(web_search_node)),
        ("response_synthesizer", RunnableLambda(response_synthesizer_node)),
        ("general_assistant", RunnableLambda(general_assistant_node)),
        ("fallback_handler", RunnableLambda(fallback_handler_node)),
        ("response_formatter", RunnableLambda(response_formatter_node)),
    ]
    
    for name, node in nodes:
        graph.add_node(name, node)
        logger.debug(f"Узел добавлен: {name}")
    
    # Установка точки входа
    graph.set_entry_point("router")
    logger.info("Точка входа установлена: router")
    
    # ------------------------------
    # Конфигурация переходов
    # ------------------------------
    
    # Маршрутизация после роутера
    graph.add_conditional_edges(
        source="router",
        path_map={
            "analytics": "schema_master",
            "documentation": "query_classifier",
            "general": "general_assistant",
            "error": "fallback_handler"
        },
        condition=lambda state: state.get("route") or "error"
    )
    logger.debug("Маршрутизация настроена")
    
    # Аналитический пайплайн
    graph.add_edge("schema_master", "sql_generator")
    graph.add_edge("sql_generator", "sql_validator")
    
    # Обработка ошибок после валидации SQL
    graph.add_conditional_edges(
        source="sql_validator",
        path_map={
            "fallback_handler": "fallback_handler",
            "access_check": "access_check"
        },
        condition=lambda state: "fallback_handler" if state.get("error") else "access_check"
    )
    
    graph.add_edge("access_check", "db_executor")
    
    # Обработка ошибок после выполнения SQL
    graph.add_conditional_edges(
        source="db_executor",
        path_map={
            "fallback_handler": "fallback_handler",
            "data_interpreter": "data_interpreter"
        },
        condition=lambda state: "fallback_handler" if state.get("error") else "data_interpreter"
    )
    
    graph.add_edge("data_interpreter", "data_storyteller")
    graph.add_edge("data_storyteller", "response_formatter")
    
    # Документационный пайплайн
    graph.add_conditional_edges(
        source="query_classifier",
        path_map={
            "small_talk": "small_talk",
            "rag_search": "rag_search",
            "web_search": "web_search"
        },
        condition=lambda state: state.get("doc_query_type") or "small_talk"
    )
    
    graph.add_edge("small_talk", "response_synthesizer")
    graph.add_edge("rag_search", "response_synthesizer")
    graph.add_edge("web_search", "response_synthesizer")
    
    # Обработка ошибок после синтеза ответа
    graph.add_conditional_edges(
        source="response_synthesizer",
        path_map={
            "fallback_handler": "fallback_handler",
            "response_formatter": "response_formatter"
        },
        condition=lambda state: "fallback_handler" if state.get("error") else "response_formatter"
    )
    
    # Прямой путь для общего ассистента
    graph.add_edge("general_assistant", "response_formatter")
    
    # Обработка ошибок в fallback_handler
    graph.add_conditional_edges(
        source="fallback_handler",
        path_map={
            "schema_master": "schema_master",  # retry
            "general_assistant": "general_assistant",  # simplify
            "response_formatter": "response_formatter"  # escalate
        },
        condition=lambda state: state.get("fallback_action") or "response_formatter"
    )
    
    # Завершающий узел
    graph.add_edge("response_formatter", END)
    
    logger.info("Конфигурация переходов завершена")
    
    # Компиляция графа
    compiled_graph = graph.compile()
    logger.info("Граф успешно скомпилирован")
    
    return compiled_graph

# Скомпилированный экземпляр графа
app = build_telegram_analytics_graph()

# ------------------------------
# Вспомогательные функции
# ------------------------------

def visualize_graph():
    """Генерирует визуальное представление графа"""
    from langgraph.graph import export_graph
    dot = export_graph(app)
    dot.save("telegram_analytics_graph.dot")
    dot.render("telegram_analytics_graph", format="png")
    logger.info("Визуализация графа сохранена в telegram_analytics_graph.png")

def get_graph():
    """Возвращает скомпилированный граф"""
    return app

def process_request(state: AgentState):
    """Обрабатывает запрос через граф"""
    logger.info(f"Начало обработки запроса: {state['user_query']}")
    try:
        # Выполняем граф
        for step in app.stream(state):
            node_name, new_state = next(iter(step.items()))
            logger.debug(f"Узел '{node_name}' завершен. Состояние: {state_snapshot(new_state)}")
        
        logger.info(f"Запрос обработан успешно. Тип ответа: {new_state.get('response_type', 'unknown')}")
        return new_state
    except Exception as e:
        logger.critical(f"Критическая ошибка при выполнении графа: {str(e)}")
        return {
            "error": "system_failure",
            "message": "Произошла критическая ошибка системы"
        }

def state_snapshot(state: AgentState) -> dict:
    """Создает сокращенный снимок состояния для логирования"""
    return {
        "route": state.get("route"),
        "error": state.get("error", {}).get("type") if state.get("error") else None,
        "sql": state.get("generated_sql", "")[:50] + "..." if state.get("generated_sql") else None,
        "result_rows": len(state["query_result"]) if isinstance(state.get("query_result"), list) else None,
        "response_type": state.get("response_type")
    }

if __name__ == "__main__":
    # Тестовый запуск и визуализация
    logging.basicConfig(level=logging.INFO)
    visualize_graph()
    
    # Пример тестового запроса
    test_state = AgentState(
        user_id="UA-123",
        chat_id=123456789,
        user_query="Покажи продажи за последний месяц"
    )
    
    result = process_request(test_state)
    print("Результат обработки:", result.get("final_response", {}))