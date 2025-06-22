from langsmith import traceable
import logging
import config
import re
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from io import BytesIO
import base64

class ResponseSynthesizer:
    def __init__(self):
        self.logger = logging.getLogger("response_synthesizer")
        self.max_text_length = 4000  # Лимит Telegram для текста
        self.max_caption_length = 1000  # Лимит подписей к медиа

    @traceable
    def synthesize(self, agent_response: dict, user_context: dict) -> dict:
        """
        Форматирует финальный ответ для отправки в Telegram.
        
        :param agent_response: Результат от исполнительного агента (DataStoryteller, GeneralAssistant и др.)
        :param user_context: Контекст пользователя (язык, настройки)
        :return: Структура {type, content, buttons}, готовая для Telegram API
        """
        try:
            # Определение типа контента
            response_type = agent_response.get("type", "text")
            
            # Обработка ошибок
            if "error" in agent_response:
                return self._format_error(agent_response)
            
            # Маршрутизация по типам ответа
            if response_type == "text":
                return self._format_text(agent_response, user_context)
            elif response_type == "image":
                return self._format_image(agent_response, user_context)
            elif response_type == "mixed":
                return self._format_mixed(agent_response, user_context)
            else:
                return self._format_fallback(agent_response)

        except Exception as e:
            self.logger.error(f"Response synthesis failed: {str(e)}")
            return {
                "type": "text",
                "content": "⚠️ Ошибка форматирования ответа. Попробуйте переформулировать запрос.",
                "buttons": []
            }

    def _format_text(self, response: dict, context: dict) -> dict:
        """Форматирует текстовый ответ с учетом лимитов Telegram"""
        text = response["content"]
        
        # Обрезаем длинные сообщения
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length - 100] + "..." 
        
        # Добавляем разметку
        formatted_text = self._apply_telegram_formatting(text, context)
        
        return {
            "type": "text",
            "content": formatted_text,
            "buttons": self._generate_buttons(response, context)
        }

    def _format_image(self, response: dict, context: dict) -> dict:
        """Форматирует ответ с изображением"""
        # Проверка base64 данных
        if not response["content"]["data"].startswith("data:image/png;base64,"):
            image_data = f"data:image/png;base64,{response['content']['data']}"
        else:
            image_data = response["content"]["data"]
        
        # Форматирование подписи
        caption = response.get("caption", "Результат анализа")
        if len(caption) > self.max_caption_length:
            caption = caption[:self.max_caption_length - 50] + "..."

        formatted_caption = self._apply_telegram_formatting(caption, context)
        
        return {
            "type": "image",
            "content": {
                "image": image_data,
                "caption": formatted_caption
            },
            "buttons": self._generate_buttons(response, context)
        }

    def _format_mixed(self, response: dict, context: dict) -> dict:
        """Обрабатывает комбинированные ответы (текст + изображение)"""
        # Telegram не позволяет отправлять текст + фото в одном сообщении
        # Разбиваем на 2 сообщения: сначала фото с подписью, затем текст
        image_part = {
            "type": "image",
            "content": response["content"]["image"],
            "caption": response.get("caption", "")
        }
        
        text_part = {
            "type": "text",
            "content": response["content"]["text"]
        }
        
        return {
            "type": "multi",
            "content": [
                self._format_image(image_part, context),
                self._format_text(text_part, context)
            ]
        }

    def _generate_buttons(self, response: dict, context: dict) -> list:
        """Генерирует интерактивные кнопки для Telegram"""
        buttons = []
        
        # 1. Кнопки действий из ответа агента
        for action in response.get("actions", []):
            buttons.append(
                InlineKeyboardButton(
                    action["title"], 
                    callback_data=f"action:{action['command']}"
                )
            )
        
        # 2. Быстрые ответы (suggestions)
        if "suggestions" in response:
            for suggestion in response["suggestions"]:
                buttons.append(
                    InlineKeyboardButton(
                        f"🔍 {suggestion}", 
                        callback_data=f"quick:{suggestion[:30]}"
                    )
                )
        
        # 3. Навигационные кнопки
        buttons.append(
            InlineKeyboardButton("📊 Новый запрос", callback_data="new_query")
        )
        
        # Группировка по 2 кнопки в ряд
        return [buttons[i:i+2] for i in range(0, len(buttons), 2)]

    def _apply_telegram_formatting(self, text: str, context: dict) -> str:
        """Применяет Telegram-разметку к тексту"""
        # Автоматическое форматирование
        formatted = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)  # Жирный -> Курсив
        formatted = re.sub(r'__(.*?)__', r'_\1_', formatted)  # Подчеркивание
        
        # Добавление эмодзи по контексту
        if "analytics" in context.get("response_type", ""):
            formatted = "📈 " + formatted
        elif "error" in context:
            formatted = "⚠️ " + formatted
        
        return formatted

    def _format_error(self, response: dict) -> dict:
        """Специальное форматирование для ошибок"""
        return {
            "type": "text",
            "content": f"🚫 Ошибка: {response.get('message', 'Неизвестная ошибка')}\n\n{response.get('details', '')}",
            "buttons": [
                [InlineKeyboardButton("🆘 Поддержка", callback_data="help")]
            ]
        }

    def _format_fallback(self, response: dict) -> dict:
        """Фолбэк для неизвестных форматов"""
        return {
            "type": "text",
            "content": str(response)[:self.max_text_length],
            "buttons": []
        }