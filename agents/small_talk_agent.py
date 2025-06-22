from langsmith import traceable
from gigachain import GigaChatModel
import requests
import logging
import config
import random
import datetime
import json
from typing import Dict, List, Optional

class SmallTalkAgent:
    def __init__(self):
        self.logger = logging.getLogger("small_talk_agent")
        self.llm = GigaChatModel(
            model="GigaChat-Plus",
            temperature=0.7,
            api_key=config.GIGACHAT_API_KEY,
            max_tokens=300
        )
        self.personality = {
            "name": "Алекс",
            "mood": "дружелюбный",
            "interests": ["технологии", "аналитика данных", "искусственный интеллект"],
            "response_style": "неформальный с юмором"
        }
        self.user_profiles = {}
        self.response_cache = {}
        
    @traceable
    def respond(self, user_query: str, user_context: Dict) -> Dict:
        """
        Обрабатывает разговорный запрос и генерирует ответ
        :param user_query: Текст запроса пользователя
        :param user_context: Контекст пользователя (имя, история, предпочтения)
        :return: Ответ в формате {"response": текст, "suggestions": [варианты], "mood": настроение}
        """
        try:
            # Обновление профиля пользователя
            self._update_user_profile(user_context)
            
            # Определение типа small talk
            talk_type = self._classify_talk_type(user_query)
            
            # Генерация ответа
            if talk_type == "greeting":
                return self._handle_greeting(user_query, user_context)
            elif talk_type == "mood":
                return self._handle_mood(user_query, user_context)
            elif talk_type == "weather":
                return self._handle_weather(user_query, user_context)
            elif talk_type == "news":
                return self._handle_news(user_query, user_context)
            elif talk_type == "personal":
                return self._handle_personal(user_query, user_context)
            elif talk_type == "about_ai":
                return self._handle_ai_questions(user_query, user_context)
            else:
                return self._handle_general(user_query, user_context)
                
        except Exception as e:
            self.logger.error(f"Small talk error: {str(e)}")
            return self._fallback_response(user_query)

    def _update_user_profile(self, context: Dict):
        """Обновляет профиль пользователя в памяти"""
        user_id = context.get("user_id")
        if not user_id:
            return
            
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "name": context.get("name", "друг"),
                "interaction_count": 0,
                "preferred_topics": [],
                "last_interaction": datetime.datetime.now().isoformat()
            }
        
        self.user_profiles[user_id]["interaction_count"] += 1
        self.user_profiles[user_id]["last_interaction"] = datetime.datetime.now().isoformat()

    def _classify_talk_type(self, query: str) -> str:
        """Классифицирует тип разговорного запроса"""
        query_lower = query.lower()
        
        greeting_words = ["привет", "здравствуй", "добрый", "хай", "здорово"]
        mood_words = ["как дела", "как жизнь", "настроение", "чувствуешь"]
        weather_words = ["погода", "дождь", "снег", "солнце", "температура"]
        news_words = ["новости", "события", "происшествия", "обновления"]
        personal_words = ["ты", "твое имя", "твои интересы", "кто тебя создал"]
        ai_words = ["ии", "искусственный интеллект", "нейросеть", "чатбот"]
        
        if any(word in query_lower for word in greeting_words):
            return "greeting"
        if any(word in query_lower for word in mood_words):
            return "mood"
        if any(word in query_lower for word in weather_words):
            return "weather"
        if any(word in query_lower for word in news_words):
            return "news"
        if any(word in query_lower for word in personal_words):
            return "personal"
        if any(word in query_lower for word in ai_words):
            return "about_ai"
            
        return "general"

    def _handle_greeting(self, query: str, context: Dict) -> Dict:
        """Обработка приветствий"""
        user_id = context.get("user_id")
        user_name = self.user_profiles.get(user_id, {}).get("name", "друг")
        
        greetings = [
            f"Привет, {user_name}! Рад тебя видеть!",
            f"Здравствуй, {user_name}! Как твои дела?",
            f"Приветствую! {self._get_time_based_greeting()}",
            f"Хай, {user_name}! Чем могу помочь?"
        ]
        
        return {
            "response": random.choice(greetings),
            "suggestions": ["Как погода?", "Что нового?", "Расскажи о себе"],
            "mood": "positive"
        }

    def _get_time_based_greeting(self) -> str:
        """Возвращает приветствие в зависимости от времени суток"""
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12:
            return "Доброе утро!"
        elif 12 <= hour < 17:
            return "Добрый день!"
        elif 17 <= hour < 23:
            return "Добрый вечер!"
        else:
            return "Доброй ночи! Ты поздно засиделся!"

    def _handle_mood(self, query: str, context: Dict) -> Dict:
        """Обработка вопросов о настроении"""
        moods = [
            "Отлично! Готов помогать с аналитикой и не только.",
            "Как у цифрового помощника - просто прекрасно!",
            "Лучше не бывает, особенно когда общаюсь с тобой!",
            "Все системы функционируют нормально, спасибо, что спросил!"
        ]
        
        # Добавляем вопрос пользователю
        follow_ups = [
            "А у тебя как настроение?",
            "Как сам себя чувствуешь?",
            "Надеюсь, у тебя тоже всё хорошо!"
        ]
        
        response = f"{random.choice(moods)} {random.choice(follow_ups)}"
        
        return {
            "response": response,
            "suggestions": ["Отлично!", "Нормально", "Не очень"],
            "mood": "positive"
        }

    def _handle_weather(self, query: str, context: Dict) -> Dict:
        """Обработка запросов о погоде"""
        # Получение погоды по умолчанию для Москвы
        location = self._extract_location(query) or "Москва"
        weather = self._get_weather(location)
        
        if weather:
            response = (f"Сейчас в {location}: {weather['description']}, "
                       f"температура {weather['temp']}°C. "
                       f"Влажность: {weather['humidity']}%.")
        else:
            response = "Извини, не могу получить данные о погоде. Попробуй посмотреть в погодном приложении!"
        
        return {
            "response": response,
            "suggestions": ["В моём городе", "На завтра", "На выходные"],
            "mood": "neutral"
        }

    def _extract_location(self, query: str) -> Optional[str]:
        """Извлекает локацию из запроса"""
        prompt = f"""Извлеки название города из запроса:
        
Запрос: "{query}"

Если город не указан - верни null. Ответ в JSON формате: {{"city": "название"}}"""
        
        response = self.llm.generate(prompt, response_format="json")
        data = json.loads(response)
        return data.get("city")

    def _get_weather(self, city: str) -> Optional[Dict]:
        """Получает текущую погоду через API (заглушка)"""
        try:
            # Реальная интеграция с OpenWeatherMap
            response = requests.get(
                f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={config.WEATHER_API_KEY}&units=metric&lang=ru"
            )
            data = response.json()
            return {
                "temp": data["main"]["temp"],
                "description": data["weather"][0]["description"],
                "humidity": data["main"]["humidity"]
            }
        except:
            # Заглушка для демо
            weather_data = {
                "Москва": {"temp": 18, "description": "легкий дождь", "humidity": 75},
                "Санкт-Петербург": {"temp": 15, "description": "облачно", "humidity": 80},
                "Новосибирск": {"temp": 22, "description": "ясно", "humidity": 45}
            }
            return weather_data.get(city, {"temp": 20, "description": "ясно", "humidity": 60})

    def _handle_news(self, query: str, context: Dict) -> Dict:
        """Обработка запросов о новостях"""
        news = self._get_news()
        return {
            "response": f"Вот последние новости:\n\n{news}",
            "suggestions": ["Технологии", "Бизнес", "Наука"],
            "mood": "informative"
        }

    def _get_news(self) -> str:
        """Получает последние новости (заглушка)"""
        try:
            # Реальная интеграция с newsapi
            response = requests.get(
                f"https://newsapi.org/v2/top-headlines?country=ru&apiKey={config.NEWS_API_KEY}"
            )
            articles = response.json().get("articles", [])[:3]
            return "\n".join([f"- {art['title']}" for art in articles])
        except:
            # Заглушка для демо
            return "1. ИИ научился предсказывать погоду с точностью 95%\n2. Компания ТехноКорп запустила новый аналитический модуль\n3. Учёные создали нейросеть для диагностики заболеваний"

    def _handle_personal(self, query: str, context: Dict) -> Dict:
        """Ответы на личные вопросы о боте"""
        user_id = context.get("user_id")
        user_name = self.user_profiles.get(user_id, {}).get("name", "друг")
        
        if "имя" in query.lower():
            response = f"Меня зовут {self.personality['name']}! А тебя?"
        elif "интерес" in query.lower():
            interests = ", ".join(self.personality['interests'][:-1]) + " и " + self.personality['interests'][-1]
            response = f"Я увлекаюсь {interests}. А что нравится тебе, {user_name}?"
        elif "создал" in query.lower():
            response = "Меня разработала команда инженеров компании ТехноКорп. Я создан, чтобы помогать с аналитикой и отвечать на вопросы!"
        else:
            response = "Я цифровой помощник, специализирующийся на аналитике данных. Чем могу тебе помочь?"
        
        return {
            "response": response,
            "suggestions": ["Расскажи о возможностях", "Что ты умеешь?", "Как работать с аналитикой"],
            "mood": "friendly"
        }

    def _handle_ai_questions(self, query: str, context: Dict) -> Dict:
        """Ответы на вопросы про ИИ"""
        prompt = f"""Ты эксперт по ИИ. Ответь на вопрос пользователя простым языком:
        
Вопрос: "{query}"

Ответ должен быть:
- Понятным для неспециалиста
- Не длиннее 3 предложений
- С примерами, если уместно
"""
        response = self.llm.generate(prompt)
        return {
            "response": response,
            "suggestions": ["Как это работает?", "Примеры использования", "Ограничения ИИ"],
            "mood": "educational"
        }

    def _handle_general(self, query: str, context: Dict) -> Dict:
        """Обработка общих разговорных запросов"""
        user_id = context.get("user_id")
        user_name = self.user_profiles.get(user_id, {}).get("name", "друг")
        
        prompt = f"""Ты дружелюбный ИИ-ассистент. Ответь на реплику пользователя:
        
Пользователь ({user_name}): "{query}"

Контекст:
- Твоё имя: {self.personality['name']}
- Твои интересы: {", ".join(self.personality['interests'])}
- Стиль общения: {self.personality['response_style']}

Ответ должен быть:
- Коротким (1-2 предложения)
- Дружелюбным
- С элементом лёгкого юмора (если уместно)
"""
        response = self.llm.generate(prompt)
        return {
            "response": response,
            "suggestions": ["Расскажи шутку", "Что нового?", "Помоги с аналитикой"],
            "mood": "friendly"
        }

    def _fallback_response(self, query: str) -> Dict:
        """Фолбэк-ответ при ошибках"""
        return {
            "response": "Извини, я немного запутался. Можешь переформулировать?",
            "suggestions": ["Привет", "Как дела?", "Что ты умеешь?"],
            "mood": "confused"
        }

    def add_custom_response(self, trigger: str, response: str):
        """Добавляет кастомный ответ на ключевые фразы"""
        # В реальной системе сохраняем в БД
        self.logger.info(f"Added custom response for: {trigger}")
        
    def learn_from_interaction(self, user_id: str, query: str, response: str, feedback: int):
        """Обучение на основе фидбека пользователя"""
        # В реальной системе сохраняем в векторную БД
        self.logger.info(f"Learning from feedback: user={user_id}, feedback={feedback}")

    def get_user_profile(self, user_id: str) -> Dict:
        """Возвращает профиль пользователя"""
        return self.user_profiles.get(user_id, {})