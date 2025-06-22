from langsmith import traceable
from gigachain import GigaChatModel
import json
import logging
import config
import time
from typing import Dict, List, Optional

class RouterAgent:
    def __init__(self, query_classifier, system_monitor):
        self.query_classifier = query_classifier
        self.system_monitor = system_monitor
        self.logger = logging.getLogger("router_agent")
        
        # Инициализация GigaChain для улучшенной маршрутизации
        self.llm = GigaChatModel(
            model="GigaChat-Pro",
            temperature=0.3,
            api_key=config.GIGACHAT_API_KEY,
            max_tokens=500
        )
        
        self.routing_rules = self._load_routing_rules()
        self.fallback_priority = ["general_assistant", "fallback_agent"]
        self.last_health_check = 0
        self.context_analysis_cache = {}
        
    @traceable
    def route(self, user_query: str, user_context: Dict, chat_history: List[Dict] = None) -> Dict:
        """
        Интеллектуальная маршрутизация запроса с учетом контекста
        
        :param user_query: Текст запроса пользователя
        :param user_context: Контекст пользователя (роль, настройки)
        :param chat_history: История диалога ([{"role": "user/assistant", "content": "текст"}])
        :return: {
            "handler": "agent_name", 
            "params": {...}, 
            "confidence": 0.0-1.0,
            "fallback_used": bool,
            "llm_enhanced": bool
        }
        """
        try:
            # Проверка системного статуса (с кешированием)
            self._check_system_health()
            
            # Базовая классификация запроса
            classification = self.query_classifier.classify_query(user_query, user_context)
            llm_enhanced = False
            
            # Уточнение классификации с помощью LLM при необходимости
            if self._needs_enhanced_routing(classification, user_context, chat_history):
                classification = self._enhance_with_llm(
                    user_query, 
                    classification, 
                    chat_history or [],
                    user_context
                )
                llm_enhanced = True
                self.logger.debug("Used LLM for enhanced routing")
            
            # Определение основного обработчика
            primary_handler = classification["handler"]
            
            # Проверка доступности обработчика
            if self._is_handler_available(primary_handler):
                return {
                    "handler": primary_handler,
                    "params": self._prepare_handler_params(
                        primary_handler, 
                        user_query, 
                        user_context,
                        chat_history,
                        classification
                    ),
                    "confidence": classification["confidence"],
                    "fallback_used": False,
                    "llm_enhanced": llm_enhanced
                }
                
            # Поиск альтернативы
            return self._find_fallback(
                primary_handler, 
                user_query, 
                user_context,
                chat_history,
                classification,
                llm_enhanced
            )
            
        except Exception as e:
            self.logger.error(f"Routing failed: {str(e)}")
            return self._emergency_fallback(user_query, user_context, chat_history)

    def _needs_enhanced_routing(
        self, 
        classification: Dict, 
        context: Dict, 
        history: List[Dict]
    ) -> bool:
        """Определяет, нужен ли углубленный LLM-анализ"""
        # Критерии для улучшенной маршрутизации:
        # 1. Низкая уверенность классификации
        if classification["confidence"] < 0.7:
            return True
            
        # 2. Сложный контекст (длинная история диалога)
        if history and len(history) > 3:
            return True
            
        # 3. Специальный флаг в контексте
        if context.get("require_llm_routing", False):
            return True
            
        # 4. Неоднозначные категории
        ambiguous_categories = ["other", "explanation", "mixed"]
        if classification["category"] in ambiguous_categories:
            return True
            
        return False

    def _enhance_with_llm(
        self, 
        query: str, 
        classification: Dict, 
        history: List[Dict],
        context: Dict
    ) -> Dict:
        """Уточняет классификацию с помощью LLM"""
        # Проверка кеша
        cache_key = f"{query[:50]}-{context.get('user_id', '')}"
        if cache_key in self.context_analysis_cache:
            return self.context_analysis_cache[cache_key]
        
        prompt = f"""Ты эксперт по маршрутизации запросов в аналитической системе. 
        
### Запрос пользователя:
{query}

### Контекст:
- Роль: {context.get('role', 'неизвестно')}
- Текущая задача: {context.get('current_task', 'не определена')}
- История диалога (последние 3 реплики):
{self._format_history(history[-3:])}

### Первоначальная классификация:
- Категория: {classification['category']}
- Уверенность: {classification['confidence']}
- Обработчик: {classification['handler']}

### Анализ:
1. Соответствует ли категория сути запроса с учетом контекста?
2. Какие нюансы истории диалога могут влиять на обработку?
3. Предложи оптимальный обработчик (analytics_pipeline, documentation_search, general_assistant, small_talk_agent, help_system)

### Требования к ответу:
- Верни JSON формата {{"final_category": "...", "final_handler": "...", "confidence": 0.0-1.0, "reason": "..."}}
- Будь лаконичным в объяснении
- Учитывай системные ограничения
"""
        
        response = self.llm.generate(prompt, response_format="json")
        
        try:
            enhanced = json.loads(response)
            # Валидация ответа
            if "final_handler" not in enhanced or "confidence" not in enhanced:
                raise ValueError("Invalid LLM response format")
                
            result = {
                **classification,
                "category": enhanced.get("final_category", classification["category"]),
                "handler": enhanced["final_handler"],
                "confidence": min(1.0, max(0.1, float(enhanced["confidence"])))
            }
            
            # Кеширование результата
            self.context_analysis_cache[cache_key] = result
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"LLM enhancement failed: {str(e)}. Using original classification")
            return classification

    def _format_history(self, history: List[Dict]) -> str:
        """Форматирует историю диалога"""
        if not history:
            return "История отсутствует"
            
        return "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in history
        ])

    def _check_system_health(self):
        """Проверяет статус систем с кешированием"""
        current_time = time.time()
        if current_time - self.last_health_check < 30:  # Кеш на 30 секунд
            return
            
        self.last_health_check = current_time
        self.system_monitor.check_all_services()

    def _is_handler_available(self, handler_name: str) -> bool:
        """Проверяет доступность обработчика"""
        # Всегда доступные обработчики
        if handler_name in ["fallback_agent", "general_assistant"]:
            return True
            
        return self.system_monitor.is_service_up(handler_name)

    def _prepare_handler_params(
        self, 
        handler: str, 
        query: str, 
        context: Dict,
        history: List[Dict],
        classification: Dict
    ) -> Dict:
        """Формирует параметры для обработчика"""
        base_params = {
            "query": query,
            "user_context": context,
            "chat_history": history,
            "classification": classification
        }
        
        # Специфичные параметры
        if handler == "analytics_pipeline":
            return {
                **base_params,
                "timeout": 30,
                "detailed": "подробн" in query.lower(),
                "priority": "high" if context.get("role") == "manager" else "normal"
            }
            
        if handler == "documentation_search":
            return {
                **base_params,
                "version": context.get("doc_version", "latest"),
                "sources": ["knowledge_base", "api_docs"]
            }
            
        if handler == "general_assistant":
            return {
                **base_params,
                "require_full_explanation": classification["category"] == "explanation"
            }
            
        return base_params

    def _find_fallback(
        self, 
        primary_handler: str, 
        query: str, 
        context: Dict,
        history: List[Dict],
        classification: Dict,
        llm_enhanced: bool
    ) -> Dict:
        """Поиск альтернативного обработчика"""
        # 1. Поиск по правилам маршрутизации
        if primary_handler in self.routing_rules:
            for alternative in self.routing_rules[primary_handler]:
                if self._is_handler_available(alternative):
                    self.logger.warning(f"Using alternative {alternative} for {primary_handler}")
                    return {
                        "handler": alternative,
                        "params": self._prepare_handler_params(
                            alternative, 
                            query, 
                            context,
                            history,
                            classification
                        ),
                        "confidence": max(0.1, classification["confidence"] - 0.2),
                        "fallback_used": True,
                        "llm_enhanced": llm_enhanced
                    }
        
        # 2. Приоритетный фолбэк
        for handler in self.fallback_priority:
            if self._is_handler_available(handler):
                self.logger.warning(f"Using priority fallback {handler}")
                return {
                    "handler": handler,
                    "params": {
                        "query": query,
                        "user_context": context,
                        "chat_history": history
                    },
                    "confidence": 0.3,
                    "fallback_used": True,
                    "llm_enhanced": llm_enhanced
                }
                
        # 3. Аварийный фолбэк
        return self._emergency_fallback(query, context, history)

    def _emergency_fallback(self, query: str, context: Dict, history: List[Dict]) -> Dict:
        """Аварийный фолбэк при полном сбое системы"""
        self.logger.critical("All systems down! Using emergency fallback")
        return {
            "handler": "fallback_agent",
            "params": {
                "query": query,
                "user_context": context,
                "chat_history": history,
                "error_details": "complete_system_failure"
            },
            "confidence": 0.1,
            "fallback_used": True,
            "llm_enhanced": False
        }

    def _load_routing_rules(self) -> Dict[str, List[str]]:
        """Загружает правила маршрутизации из конфигурации"""
        return config.ROUTING_RULES

    def add_dynamic_rule(self, source_handler: str, target_handler: str):
        """Динамически добавляет правило маршрутизации"""
        if source_handler not in self.routing_rules:
            self.routing_rules[source_handler] = []
            
        if target_handler not in self.routing_rules[source_handler]:
            self.routing_rules[source_handler].append(target_handler)
            self.logger.info(f"Added dynamic route: {source_handler} -> {target_handler}")

    def update_fallback_priority(self, new_priority: List[str]):
        """Обновляет приоритет фолбэк-обработчиков"""
        self.fallback_priority = new_priority
        self.logger.info(f"Fallback priority updated: {new_priority}")

    def clear_cache(self):
        """Очищает кеш контекстного анализа"""
        self.context_analysis_cache = {}
        self.logger.info("Router context cache cleared")