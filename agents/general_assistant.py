from langsmith import traceable
from gigachain import GigaChatModel
import requests
import json
import logging
import config
import re
import datetime
from typing import List, Dict

class GeneralAssistant:
    def __init__(self):
        self.llm = GigaChatModel(
            model="GigaChat-Plus",
            temperature=0.7,
            api_key=config.GIGACHAT_API_KEY,
            max_tokens=500
        )
        self.logger = logging.getLogger("general_assistant")
        self.context_memory = {}
        self.personality = {
            "name": "Алексей",
            "role": "Цифровой помощник",
            "traits": "дружелюбный, вежливый, немного юмористический",
            "knowledge": "эксперт в аналитике данных и бизнес-процессах компании"
        }
        self.system_context = {
            "current_date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "company": "ТехноКорп",
            "system_version": "2.3.1"
        }
        
    @traceable
    def respond(self, user_query: str, chat_history: List[Dict] = None, user_context: Dict = None) -> Dict:
        """
        Основной метод обработки запросов пользователя
        :param user_query: Текущий запрос пользователя
        :param chat_history: История диалога в формате [{"role": "user/assistant", "content": "текст"}]
        :param user_context: Контекст пользователя (роли, предпочтения и т.д.)
        :return: Ответ помощника
        """
        try:
            # Определяем тип запроса
            query_type = self._classify_query(user_query)
            
            # Обработка в зависимости от типа
            if query_type == "greeting":
                return self._handle_greeting(user_query)
            elif query_type == "system_info":
                return self._handle_system_info(user_query)
            elif query_type == "small_talk":
                return self._handle_small_talk(user_query)
            elif query_type == "contextual":
                return self._handle_contextual(user_query, chat_history, user_context)
            elif query_type == "complex_task":
                return self._handle_complex_task(user_query, chat_history, user_context)
            else:
                return self._fallback_response(user_query)
                
        except Exception as e:
            self.logger.error(f"General assistant error: {str(e)}")
            return self._error_response(user_query)

    def _classify_query(self, query: str) -> str:
        """Классифицирует тип запроса"""
        query_lower = query.lower()
        
        # Проверка приветствий
        if re.search(r"привет|здравствуй|добрый|хай|здорово", query_lower):
            return "greeting"
        
        # Проверка системных запросов
        if re.search(r"версия|обнов|дата релиза|сборк|статус", query_lower):
            return "system_info"
        
        # Проверка разговорных запросов
        if re.search(r"как дела|как жизнь|настроени|погод|новост", query_lower):
            return "small_talk"
        
        # Проверка контекстных запросов
        if re.search(r"напомни|что говорил|ранее|в прошлый раз", query_lower):
            return "contextual"
        
        # Проверка сложных запросов
        if len(query.split()) > 10 or re.search(r"объясни|расскажи|сравни|проанализируй", query_lower):
            return "complex_task"
        
        return "unknown"

    def _handle_greeting(self, query: str) -> Dict:
        """Обработка приветственных сообщений"""
        greetings = [
            "Привет! Чем могу помочь?",
            "Здравствуйте! Готов помочь с аналитикой и не только.",
            "Приветствую! Как ваши дела сегодня?",
            "Рад вас видеть! Что вас интересует?",
            "Добрый день! Чем займемся сегодня?"
        ]
        
        # Выбор случайного приветствия
        response = np.random.choice(greetings)
        
        # Добавляем имя, если оно есть в запросе
        name_match = re.search(r"(?:меня зовут|я) (\w+)", query, re.IGNORECASE)
        if name_match:
            name = name_match.group(1)
            response = f"{response.split('!')[0]}, {name}! {response.split('!')[1]}"
            self.context_memory["user_name"] = name
        
        return {
            "type": "text",
            "content": response,
            "suggestions": [
                "Покажи последние продажи",
                "Как создать отчет?",
                "Что ты умеешь?"
            ]
        }

    def _handle_system_info(self, query: str) -> Dict:
        """Обработка запросов о системе"""
        if re.search(r"версия|версии", query.lower()):
            response = f"Текущая версия системы: {self.system_context['system_version']}"
        elif re.search(r"обнов|новое", query.lower()):
            response = "Последнее обновление было вчера. Добавлена поддержка новых отчетов по клиентской аналитике."
        elif re.search(r"дата|число|день", query.lower()):
            response = f"Сегодня {self.system_context['current_date']}"
        else:
            response = "Система работает в штатном режиме. Все компоненты функционируют нормально."
        
        return {
            "type": "text",
            "content": response,
            "system_info": True
        }

    def _handle_small_talk(self, query: str) -> Dict:
        """Обработка разговорных запросов"""
        # Специальные случаи
        if re.search(r"как дела|как жизнь", query.lower()):
            responses = [
                "У меня всё отлично, работаю на полную мощность!",
                "Как у цифрового помощника - прекрасно! Готов помогать вам.",
                "Лучше не бывает, особенно когда могу помочь вам!",
                "Все системы функционируют нормально, спасибо, что спросили!"
            ]
            response = np.random.choice(responses)
        
        elif re.search(r"погод", query.lower()):
            # Интеграция с внешним API погоды
            weather = self._get_weather()
            response = f"Сейчас {weather['description']}, температура {weather['temp']}°C"
        
        elif re.search(r"новост", query.lower()):
            # Получение последних новостей компании
            news = self._get_company_news()
            response = f"Последние новости компании:\n{news}"
        
        else:
            response = "Извините, я пока не научился обсуждать такие темы. Могу помочь с аналитикой или документацией!"
        
        return {
            "type": "text",
            "content": response,
            "small_talk": True
        }

    def _handle_contextual(self, query: str, chat_history: List[Dict], user_context: Dict) -> Dict:
        """Обработка контекстных запросов"""
        # Извлечение контекста из истории
        context = self._extract_context(chat_history, user_context)
        
        prompt = f"""Ты интеллектуальный помощник. Ответь на запрос пользователя, используя контекст:
        
        Контекст:
        - Имя пользователя: {context.get('user_name', 'неизвестно')}
        - Роль: {context.get('role', 'пользователь')}
        - История диалога (последние 3 реплики):
          {self._format_chat_history(chat_history[-3:])}
        - Системная информация: версия {self.system_context['system_version']}, дата {self.system_context['current_date']}
        
        Запрос пользователя: {query}
        
        Ответ должен быть:
        - Максимально полезным
        - Учитывать историю диалога
        - Вежливым и дружелюбным
        - Не более 3 предложений
        """
        
        response = self.llm.generate(prompt)
        return {
            "type": "text",
            "content": response,
            "context_used": True
        }

    def _handle_complex_task(self, query: str, chat_history: List[Dict], user_context: Dict) -> Dict:
        """Обработка сложных аналитических и объясняющих запросов"""
        # Извлечение контекста
        context = self._extract_context(chat_history, user_context)
        
        prompt = f"""Ты старший аналитик компании {self.system_context['company']}. 
        Ответь на сложный запрос пользователя, следуя инструкциям:
        
        Информация о пользователе:
        - Имя: {context.get('user_name', 'неизвестно')}
        - Роль: {context.get('role', 'пользователь')}
        - Уровень знаний: {context.get('expertise', 'средний')}
        
        Запрос: {query}
        
        Структура ответа:
        1. Краткое резюме сути запроса
        2. Пошаговое объяснение/решение
        3. Ключевые выводы
        4. Рекомендации (если применимо)
        5. Дополнительные ресурсы (если нужны)
        
        Используй профессиональный, но доступный язык. Допустимы маркированные списки.
        """
        
        response = self.llm.generate(prompt)
        
        # Пост-обработка для улучшения читаемости
        formatted_response = self._format_complex_response(response)
        
        return {
            "type": "text",
            "content": formatted_response,
            "complex_task": True,
            "suggestions": ["Могу ли я что-то еще объяснить?", "Нужна дополнительная помощь?"]
        }

    def _extract_context(self, chat_history: List[Dict], user_context: Dict) -> Dict:
        """Извлекает контекст из истории и данных пользователя"""
        context = {
            "user_name": self.context_memory.get("user_name", "пользователь"),
            "role": user_context.get("role", "сотрудник"),
            "expertise": user_context.get("expertise_level", "средний")
        }
        
        # Извлечение тем из истории
        if chat_history:
            topics = self._extract_topics(chat_history)
            context["recent_topics"] = topics[:3]
            
        # Сохранение в памяти
        self.context_memory.update(context)
        return context

    def _extract_topics(self, chat_history: List[Dict]) -> List[str]:
        """Извлекает основные темы из истории диалога"""
        history_text = "\n".join([msg["content"] for msg in chat_history])
        
        prompt = f"""Определи основные темы в истории диалога:
        
        История:
        {history_text}
        
        Выведи только список тем через запятую, без дополнительного текста.
        """
        
        topics = self.llm.generate(prompt)
        return [t.strip() for t in topics.split(",") if t.strip()]

    def _format_chat_history(self, history: List[Dict]) -> str:
        """Форматирует историю диалога для промпта"""
        return "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in history
        ])

    def _format_complex_response(self, response: str) -> str:
        """Улучшает форматирование сложных ответов"""
        # Добавление маркированных списков
        response = re.sub(r"\n\d+\.\s", "\n• ", response)
        
        # Выделение ключевых терминов
        key_terms = ["Важно:", "Ключевое:", "Рекомендация:"]
        for term in key_terms:
            response = response.replace(term, f"**{term}**")
        
        # Упрощение длинных абзацев
        sentences = response.split(". ")
        if len(sentences) > 5:
            response = ". ".join(sentences[:5]) + ".\n\n[Ответ сокращен для удобства]"
            
        return response

    def _get_weather(self) -> Dict:
        """Получает текущую погоду (заглушка с реальной интеграцией)"""
        try:
            # Реальная интеграция с OpenWeatherMap
            response = requests.get(
                f"http://api.openweathermap.org/data/2.5/weather?q=Moscow&appid={config.WEATHER_API_KEY}&units=metric&lang=ru"
            )
            data = response.json()
            return {
                "temp": data["main"]["temp"],
                "description": data["weather"][0]["description"]
            }
        except:
            return {"temp": 20, "description": "ясно"}

    def _get_company_news(self) -> str:
        """Получает последние новости компании (заглушка)"""
        return "1. Запущена новая система аналитики\n2. Компания получила награду 'Лучший работодатель года'\n3. Запланировано обновление на следующей неделе"

    def _fallback_response(self, query: str) -> Dict:
        """Ответ по умолчанию для неизвестных запросов"""
        prompt = f"""Пользователь спросил: "{query}"
        
        Ты не знаешь точного ответа, но хочешь помочь. Предложи:
        1. Альтернативные формулировки вопроса
        2. Связанные темы, которые ты знаешь
        3. Возможность перенаправить запрос специалисту
        
        Ответ должен быть вежливым и полезным.
        """
        
        response = self.llm.generate(prompt)
        return {
            "type": "text",
            "content": response,
            "fallback": True,
            "suggestions": [
                "Попробуй задать вопрос по-другому",
                "Могу помочь с аналитикой данных",
                "Хочешь, я перенаправлю твой вопрос специалисту?"
            ]
        }

    def _error_response(self, query: str) -> Dict:
        """Ответ при внутренних ошибках"""
        return {
            "type": "text",
            "content": (
                "Кажется, у меня небольшие технические трудности. "
                "Пожалуйста, повторите запрос чуть позже или попробуйте переформулировать."
            ),
            "error": True
        }

    def update_context(self, user_id: str, key: str, value: any):
        """Обновляет контекст пользователя"""
        if user_id not in self.context_memory:
            self.context_memory[user_id] = {}
        self.context_memory[user_id][key] = value

    def get_context(self, user_id: str) -> Dict:
        """Возвращает сохраненный контекст пользователя"""
        return self.context_memory.get(user_id, {})