from langsmith import traceable
from gigachain import GigaChatModel
import requests
import json
import logging
import config
import time

class FallbackAgent:
    def __init__(self):
        self.llm = GigaChatModel(
            model="GigaChat-Pro",
            temperature=0.3,
            api_key=config.GIGACHAT_API_KEY
        )
        self.logger = logging.getLogger("fallback_agent")
        self.error_history = {}
        self.strategy_counter = {}
        
    @traceable
    def handle_error(self, error_data: dict, original_request: dict) -> dict:
        """
        Обрабатывает ошибки в системе и определяет стратегию восстановления
        :param error_data: Данные об ошибке
        :param original_request: Оригинальный запрос пользователя
        :return: Решение по обработке ошибки
        """
        try:
            # Анализ ошибки
            error_type = self._classify_error(error_data)
            error_id = self._generate_error_id(error_data)
            
            # Проверка на повторяющиеся ошибки
            if self._is_recurring_error(error_id):
                return self._handle_recurring_error(error_id, error_data, original_request)
            
            # Определение стратегии
            strategy = self._determine_strategy(error_type, error_data, original_request)
            
            # Регистрация ошибки
            self._register_error(error_id, error_data, strategy)
            
            # Выполнение стратегии
            return self._execute_strategy(strategy, error_data, original_request)
            
        except Exception as e:
            self.logger.critical(f"Fallback failure: {str(e)}")
            return self._critical_fallback(original_request)

    def _classify_error(self, error_data: dict) -> str:
        """Классифицирует тип ошибки"""
        error_msg = error_data.get("message", "").lower()
        
        # Определение типа ошибки по ключевым словам
        if "sql" in error_msg or "syntax" in error_msg:
            return "sql_error"
        elif "timeout" in error_msg or "timed out" in error_msg:
            return "timeout"
        elif "connection" in error_msg or "network" in error_msg:
            return "network_error"
        elif "auth" in error_msg or "access" in error_msg or "permission" in error_msg:
            return "access_denied"
        elif "not found" in error_msg or "no data" in error_msg:
            return "data_not_found"
        elif "validation" in error_msg or "invalid" in error_msg:
            return "validation_error"
        else:
            return "unknown_error"

    def _generate_error_id(self, error_data: dict) -> str:
        """Генерирует уникальный ID ошибки"""
        error_type = self._classify_error(error_data)
        component = error_data.get("component", "unknown")
        timestamp = int(time.time())
        return f"ERR-{error_type[:3]}-{component[:3]}-{timestamp}"

    def _is_recurring_error(self, error_id: str) -> bool:
        """Проверяет, повторяется ли ошибка"""
        # Упрощенная проверка по базовому паттерну (без полного ID)
        error_pattern = error_id.split('-')[:3]
        pattern_key = '-'.join(error_pattern)
        
        # Счетчик повторений
        count = self.strategy_counter.get(pattern_key, 0)
        return count >= 2  # Если ошибка повторилась более 2 раз

    def _determine_strategy(self, error_type: str, error_data: dict, request: dict) -> str:
        """Определяет стратегию восстановления"""
        # Простые правила для частых ошибок
        if error_type == "timeout":
            return "retry"
        elif error_type == "network_error":
            return "retry_after_delay"
        elif error_type == "access_denied":
            return "escalate"
        elif error_type == "data_not_found":
            return "simplify_query"
        
        # Для сложных случаев используем LLM
        prompt = f"""Система обработки запросов столкнулась с ошибкой. Определи лучшую стратегию восстановления.

Детали ошибки:
- Тип: {error_type}
- Компонент: {error_data.get('component', 'unknown')}
- Сообщение: {error_data.get('message', 'Нет дополнительной информации')}
- Код статуса: {error_data.get('status_code', 'N/A')}

Оригинальный запрос пользователя:
{request.get('text', '')}

Доступные стратегии:
1. retry - Повторить операцию немедленно
2. retry_after_delay - Повторить после паузы (5-15 сек)
3. simplify_query - Упростить запрос пользователя
4. alternative_approach - Попробовать альтернативный метод
5. escalate_to_human - Эскалировать в техническую поддержку
6. inform_user - Сообщить пользователю об ошибке

Выбери наиболее подходящую стратегию. Ответ должен содержать только название стратегии.
"""
        
        response = self.llm.generate(prompt)
        return response.strip().lower()

    def _execute_strategy(self, strategy: str, error_data: dict, request: dict) -> dict:
        """Выполняет выбранную стратегию"""
        if strategy == "retry":
            return self._retry_strategy(error_data, request)
        elif strategy == "retry_after_delay":
            return self._delayed_retry_strategy(error_data, request)
        elif strategy == "simplify_query":
            return self._simplify_query_strategy(error_data, request)
        elif strategy == "alternative_approach":
            return self._alternative_approach_strategy(error_data, request)
        elif strategy == "escalate_to_human":
            return self._escalation_strategy(error_data, request)
        else:
            return self._inform_user_strategy(error_data, request)

    def _retry_strategy(self, error_data: dict, request: dict) -> dict:
        """Стратегия немедленного повтора"""
        component = error_data.get("component")
        self.logger.info(f"Retrying operation for {component}")
        
        return {
            "strategy": "retry",
            "action": {
                "type": "retry_operation",
                "component": component,
                "immediately": True,
                "max_attempts": 3
            },
            "user_message": None  # Не сообщаем пользователю о повторе
        }

    def _delayed_retry_strategy(self, error_data: dict, request: dict) -> dict:
        """Стратегия повтора с задержкой"""
        delay = 10  # секунд
        self.logger.info(f"Scheduling retry after {delay} seconds")
        
        return {
            "strategy": "retry_after_delay",
            "action": {
                "type": "retry_operation",
                "component": error_data.get("component"),
                "delay": delay,
                "max_attempts": 2
            },
            "user_message": "Система временно перегружена. Пожалуйста, подождите..."
        }

    def _simplify_query_strategy(self, error_data: dict, request: dict) -> dict:
        """Стратегия упрощения запроса"""
        original_query = request.get("text", "")
        
        prompt = f"""Упрости запрос пользователя для избежания ошибки:
        
Оригинальный запрос: {original_query}
Ошибка: {error_data.get('message', 'Нет дополнительной информации')}

Упрощенный запрос должен:
1. Сохранить основную суть оригинального запроса
2. Быть более конкретным
3. Избегать сложных конструкций
4. Использовать более простые формулировки

Упрощенный запрос:"""
        
        simplified_query = self.llm.generate(prompt)
        
        return {
            "strategy": "simplify_query",
            "action": {
                "type": "process_new_query",
                "query": simplified_query
            },
            "user_message": f"Попробуем упрощенный запрос: {simplified_query}"
        }

    def _alternative_approach_strategy(self, error_data: dict, request: dict) -> dict:
        """Стратегия альтернативного подхода"""
        component = error_data.get("component")
        original_query = request.get("text", "")
        
        prompt = f"""Для запроса пользователя возникла ошибка в компоненте {component}. 
Предложи альтернативный подход к обработке запроса без использования этого компонента.

Оригинальный запрос: {original_query}
Ошибка: {error_data.get('message')}

Альтернативный подход:"""
        
        approach = self.llm.generate(prompt)
        
        return {
            "strategy": "alternative_approach",
            "action": {
                "type": "alternative_processing",
                "approach_description": approach
            },
            "user_message": "Используем альтернативный метод обработки вашего запроса"
        }

    def _escalation_strategy(self, error_data: dict, request: dict) -> dict:
        """Стратегия эскалации к человеку"""
        # Создаем тикет в Jira
        ticket_id = self._create_jira_ticket(error_data, request)
        
        # Отправляем уведомление в Slack
        self._notify_slack(error_data, ticket_id)
        
        return {
            "strategy": "escalate_to_human",
            "action": {
                "type": "create_ticket",
                "ticket_id": ticket_id,
                "system": "Jira"
            },
            "user_message": (
                "Произошла сложная ошибка. Наша команда уже уведомлена. "
                f"Тикет: #{ticket_id}. Приносим извинения за неудобства!"
            )
        }

    def _inform_user_strategy(self, error_data: dict, request: dict) -> dict:
        """Стратегия информирования пользователя"""
        error_type = self._classify_error(error_data)
        
        # Генерация понятного сообщения об ошибке
        prompt = f"""Создай понятное сообщение об ошибке для пользователя:
        
Тип ошибки: {error_type}
Техническое сообщение: {error_data.get('message', 'Нет дополнительной информации')}

Требования к сообщению:
1. Будь вежливым и извинись
2. Объясни суть проблемы простым языком
3. Предложи возможные решения
4. Сообщение должно быть не длиннее 2 предложений

Сообщение пользователю:"""
        
        user_message = self.llm.generate(prompt)
        
        return {
            "strategy": "inform_user",
            "action": {
                "type": "direct_response"
            },
            "user_message": user_message
        }

    def _register_error(self, error_id: str, error_data: dict, strategy: str):
        """Регистрирует ошибку в истории"""
        # Упрощенный ключ для группировки похожих ошибок
        pattern_key = '-'.join(error_id.split('-')[:3])
        
        # Обновляем счетчик
        self.strategy_counter[pattern_key] = self.strategy_counter.get(pattern_key, 0) + 1
        
        # Сохраняем детали ошибки
        self.error_history[error_id] = {
            "timestamp": int(time.time()),
            "error_data": error_data,
            "strategy": strategy,
            "resolved": False
        }
        
        self.logger.warning(f"Error registered: {error_id}, strategy: {strategy}")

    def _create_jira_ticket(self, error_data: dict, request: dict) -> str:
        """Создает тикет в Jira"""
        try:
            # Формирование описания ошибки
            prompt = f"""Создай техническое описание ошибки для тикета в Jira:
            
Ошибка в компоненте: {error_data.get('component', 'unknown')}
Тип ошибки: {self._classify_error(error_data)}
Сообщение: {error_data.get('message', 'Нет сообщения')}
Статус код: {error_data.get('status_code', 'N/A')}

Запрос пользователя: {request.get('text', '')}

Включи:
1. Подробное описание проблемы
2. Шаги для воспроизведения (если применимо)
3. Предполагаемую причину
4. Связанные системные логи (если есть)
"""
            description = self.llm.generate(prompt)
            
            payload = {
                "fields": {
                    "project": {"key": config.JIRA_PROJECT_KEY},
                    "summary": f"[AI System] {error_data.get('component')} Error: {self._classify_error(error_data)}",
                    "description": description,
                    "issuetype": {"name": "Bug"},
                    "priority": {"name": "High"},
                    "labels": ["ai_system", "fallback"]
                }
            }
            
            response = requests.post(
                f"{config.JIRA_BASE_URL}/rest/api/2/issue",
                json=payload,
                auth=(config.JIRA_USER, config.JIRA_API_TOKEN),
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 201:
                return response.json().get("key", "UNKNOWN-001")
            else:
                self.logger.error(f"Jira create failed: {response.text}")
                return "FAILED-" + str(int(time.time()))
                
        except Exception as e:
            self.logger.error(f"Jira integration error: {str(e)}")
            return "ERROR-" + str(int(time.time()))

    def _notify_slack(self, error_data: dict, ticket_id: str):
        """Отправляет уведомление в Slack"""
        try:
            message = {
                "text": f":fire: *Critical System Error* :fire:",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*{error_data.get('component', 'Unknown')}* encountered an error!"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Type:*\n{self._classify_error(error_data)}"},
                            {"type": "mrkdwn", "text": f"*Ticket:*\n{ticket_id}"},
                            {"type": "mrkdwn", "text": f"*Message:*\n{error_data.get('message', 'No details')}"}
                        ]
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "View in Jira"
                                },
                                "url": f"{config.JIRA_BASE_URL}/browse/{ticket_id}"
                            }
                        ]
                    }
                ]
            }
            
            requests.post(config.SLACK_WEBHOOK_URL, json=message)
            
        except Exception as e:
            self.logger.error(f"Slack notification failed: {str(e)}")

    def _handle_recurring_error(self, error_id: str, error_data: dict, request: dict) -> dict:
        """Обрабатывает повторяющиеся ошибки"""
        self.logger.error(f"Recurring error detected: {error_id}")
        
        # Эскалация в Jira с высоким приоритетом
        ticket_id = self._create_jira_ticket(error_data, request)
        self._notify_slack({
            **error_data,
            "message": f"RECURRING ERROR: {error_data.get('message')}"
        }, ticket_id)
        
        return {
            "strategy": "escalate_recurring",
            "action": {
                "type": "create_high_priority_ticket",
                "ticket_id": ticket_id
            },
            "user_message": (
                "Обнаружена повторяющаяся ошибка. Наша команда работает над решением. "
                f"Тикет: #{ticket_id}. Приносим извинения за неудобства!"
            )
        }

    def _critical_fallback(self, request: dict) -> dict:
        """Аварийный фолбэк при сбое самого FallbackAgent"""
        return {
            "strategy": "critical_fallback",
            "action": {
                "type": "direct_response"
            },
            "user_message": (
                "Произошла критическая ошибка в системе. "
                "Наша команда уже уведомлена. Пожалуйста, повторите запрос позже."
            )
        }