from langsmith import traceable
from gigachain import GigaChatModel
import pandas as pd
import numpy as np
import logging
import config

class DataInterpreterAgent:
    def __init__(self):
        self.llm = GigaChatModel(
            model="GigaChat-Pro",
            temperature=0.3,
            api_key=config.GIGACHAT_API_KEY
        )
        self.logger = logging.getLogger("data_interpreter")
        
    @traceable  # Интеграция с LangSmith
    def analyze(self, df: pd.DataFrame, user_query: str, context: dict) -> dict:
        """
        Анализирует данные DataFrame и генерирует аналитические выводы
        :param df: DataFrame с результатами SQL-запроса
        :param user_query: Оригинальный запрос пользователя
        :param context: Контекст анализа (схема БД, история и т.д.)
        :return: Словарь с результатами анализа
        """
        try:
            # Проверка на пустые данные
            if df.empty:
                return self._handle_empty_data(user_query)
            
            # Определение типа анализа
            analysis_type = self._determine_analysis_type(df, user_query)
            
            # Выбор стратегии анализа
            if analysis_type == "simple":
                return self._simple_analysis(df, user_query)
            elif analysis_type == "trend":
                return self._trend_analysis(df, user_query)
            elif analysis_type == "comparison":
                return self._comparison_analysis(df, user_query)
            elif analysis_type == "anomaly":
                return self._anomaly_detection(df, user_query)
            else:
                return self._advanced_analysis(df, user_query, context)
                
        except Exception as e:
            self.logger.error(f"Analysis error: {str(e)}")
            return {
                "error": "analysis_failed",
                "details": str(e),
                "suggestion": "Попробуйте уточнить запрос или обратитесь к администратору"
            }

    def _determine_analysis_type(self, df: pd.DataFrame, query: str) -> str:
        """Определяет тип анализа на основе запроса и данных"""
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ["тренд", "динамика", "изменение"]):
            return "trend"
        elif any(kw in query_lower for kw in ["сравнен", "vs", "против"]):
            return "comparison"
        elif any(kw in query_lower for kw in ["аномал", "отклонен", "выброс"]):
            return "anomaly"
        elif len(df) <= 5 or len(df.columns) <= 2:
            return "simple"
        else:
            return "advanced"

    def _simple_analysis(self, df: pd.DataFrame, query: str) -> dict:
        """Простой анализ для небольших наборов данных"""
        insights = []
        
        # Основные статистики для числовых колонок
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            stats = {
                "min": df[col].min(),
                "max": df[col].max(),
                "mean": df[col].mean(),
                "median": df[col].median()
            }
            insights.append(f"**{col}**: min={stats['min']}, max={stats['max']}, avg={stats['mean']:.2f}")
        
        # Форматирование результатов
        response = f"Результаты по запросу '{query}':\n"
        response += "\n".join(insights)
        
        return {
            "analysis_type": "simple",
            "content": response,
            "recommended_visualization": "table" if len(df) < 10 else "bar_chart"
        }

    def _trend_analysis(self, df: pd.DataFrame, query: str) -> dict:
        """Анализ временных трендов"""
        # Ищем колонку с датой
        date_col = next((col for col in df.columns if "date" in col.lower()), None)
        
        if not date_col:
            return self._fallback_analysis(df, query, "Не найдена колонка с датами")
        
        # Группировка по временным периодам
        df['period'] = df[date_col].dt.to_period('M')
        grouped = df.groupby('period').sum(numeric_only=True)
        
        # Формируем промпт для LLM
        prompt = f"""Проанализируй временной тренд на основе данных:
        
        Данные (агрегированные по месяцам):
        {grouped.head().to_markdown()}
        
        Запрос пользователя: {query}
        
        Сформулируй:
        1. Основной тренд
        2. Ключевые изменения
        3. Необычные наблюдения
        """
        
        analysis = self.llm.generate(prompt)
        
        return {
            "analysis_type": "trend",
            "content": analysis,
            "data_sample": grouped.head().to_dict(),
            "recommended_visualization": "line_chart"
        }

    def _comparison_analysis(self, df: pd.DataFrame, query: str) -> dict:
        """Сравнительный анализ"""
        prompt = f"""Проведи сравнительный анализ данных:
        
        Первые 5 строк данных:
        {df.head().to_markdown()}
        
        Полный размер данных: {len(df)} строк, {len(df.columns)} колонок
        
        Запрос пользователя: {query}
        
        Сравни основные показатели и выдели ключевые различия.
        """
        
        analysis = self.llm.generate(prompt)
        
        return {
            "analysis_type": "comparison",
            "content": analysis,
            "recommended_visualization": "bar_chart"
        }

    def _anomaly_detection(self, df: pd.DataFrame, query: str) -> dict:
        """Обнаружение аномалий в данных"""
        # Простой метод обнаружения выбросов
        anomalies = []
        num_cols = df.select_dtypes(include=np.number).columns
        
        for col in num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = df[(df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))]
            
            if not outliers.empty:
                anomalies.append(f"**{col}**: {len(outliers)} выбросов")
        
        # Формируем отчет
        if not anomalies:
            content = f"По запросу '{query}' значительных аномалий не обнаружено"
        else:
            content = f"Обнаружены аномалии:\n" + "\n".join(anomalies)
        
        return {
            "analysis_type": "anomaly",
            "content": content,
            "anomaly_count": len(anomalies),
            "recommended_visualization": "scatter_plot"
        }

    def _advanced_analysis(self, df: pd.DataFrame, query: str, context: dict) -> dict:
        """Расширенный анализ с использованием LLM"""
        prompt = f"""Ты senior data analyst. Проанализируй данные и ответь на вопрос.
        
        Контекст:
        - Схема БД: {context.get('schema', 'N/A')}
        - Запрос: {query}
        
        Данные (первые 5 строк):
        {df.head().to_markdown()}
        
        Технические характеристики данных:
        - Всего строк: {len(df)}
        - Всего колонок: {len(df.columns)}
        - Типы данных: {df.dtypes.to_dict()}
        
        Задание:
        1. Сформулируй ключевые инсайты
        2. Выяви скрытые закономерности
        3. Предложи рекомендации (если применимо)
        4. Определи оптимальный способ визуализации
        """
        
        analysis = self.llm.generate(prompt)
        
        # Извлекаем рекомендации по визуализации
        visualization = "bar_chart"
        if "линейный" in analysis.lower():
            visualization = "line_chart"
        elif "кругл" in analysis.lower():
            visualization = "pie_chart"
        elif "рассеян" in analysis.lower():
            visualization = "scatter_plot"
        
        return {
            "analysis_type": "advanced",
            "content": analysis,
            "recommended_visualization": visualization,
            "data_sample": df.head(3).to_dict()
        }

    def _handle_empty_data(self, query: str) -> dict:
        """Обработка случая с пустыми данными"""
        return {
            "analysis_type": "empty",
            "content": f"По вашему запросу '{query}' данные не найдены",
            "recommendation": "Попробуйте изменить параметры запроса или период",
            "recommended_visualization": None
        }

    def _fallback_analysis(self, df: pd.DataFrame, query: str, reason: str) -> dict:
        """Фолбэк-анализ при проблемах"""
        prompt = f"""Пользователь запросил: {query}
        Но возникла проблема: {reason}
        
        Данные:
        {df.head().to_markdown()}
        
        Предложи альтернативный анализ или уточняющий вопрос."""
        
        response = self.llm.generate(prompt)
        return {
            "analysis_type": "fallback",
            "content": response,
            "reason": reason
        }