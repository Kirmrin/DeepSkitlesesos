from langsmith import traceable
from gigachain import GigaChatModel
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
import logging
import config
import re

class DataStorytellerAgent:
    def __init__(self):
        self.llm = GigaChatModel(
            model="GigaChat-Pro",
            temperature=0.7,
            api_key=config.GIGACHAT_API_KEY
        )
        self.logger = logging.getLogger("data_storyteller")
        self.color_palette = sns.color_palette("viridis")
        plt.style.use("seaborn-whitegrid")
        
    @traceable
    def create_story(self, analysis_result: dict, df: pd.DataFrame, user_query: str) -> dict:
        """
        Создает визуальную историю на основе результатов анализа
        :param analysis_result: Результат от DataInterpreterAgent
        :param df: Исходный DataFrame с данными
        :param user_query: Оригинальный запрос пользователя
        :return: Словарь с текстом истории и визуализациями
        """
        try:
            # Определяем тип визуализации
            vis_type = analysis_result.get("recommended_visualization", "auto")
            if vis_type == "auto":
                vis_type = self._determine_visualization_type(df, analysis_result["analysis_type"])
            
            # Генерируем историю
            narrative = self._generate_narrative(analysis_result, user_query)
            
            # Создаем визуализацию
            visualization = self._create_visualization(df, vis_type, analysis_result)
            
            return {
                "narrative": narrative,
                "visualization": visualization,
                "visualization_type": vis_type,
                "analysis_type": analysis_result["analysis_type"],
                "data_sample": df.head(3).to_dict() if not df.empty else {}
            }
        except Exception as e:
            self.logger.error(f"Story creation failed: {str(e)}")
            return {
                "error": "story_error",
                "message": "Не удалось создать визуальную историю",
                "details": str(e),
                "fallback_narrative": analysis_result.get("content", "")
            }

    def _determine_visualization_type(self, df: pd.DataFrame, analysis_type: str) -> str:
        """Определяет оптимальный тип визуализации"""
        num_cols = len(df.select_dtypes(include=np.number).columns)
        cat_cols = len(df.select_dtypes(include='category').columns)
        
        if analysis_type == "trend":
            return "line_chart"
        elif analysis_type == "comparison":
            if cat_cols > 0 and num_cols > 0:
                return "bar_chart"
            return "scatter_plot"
        elif analysis_type == "anomaly":
            return "scatter_plot"
        elif analysis_type == "distribution":
            if cat_cols == 1:
                return "pie_chart"
            return "histogram"
        
        # По умолчанию для сложных случаев
        return "summary_table" if len(df) < 10 else "bar_chart"

    def _generate_narrative(self, analysis: dict, user_query: str) -> str:
        """Генерирует текстовое повествование на основе анализа"""
        prompt = f"""Ты data storyteller. Создай понятную историю на основе анализа данных.
        
        Результаты анализа:
        {analysis['content']}
        
        Контекст:
        - Оригинальный запрос: "{user_query}"
        - Тип анализа: {analysis['analysis_type']}
        
        Требования к истории:
        1. Начни с ключевого вывода
        2. Используй простой язык без технического жаргона
        3. Дай объяснение обнаруженным закономерностям
        4. Предложи рекомендации (если применимо)
        5. Ограничься 3-5 предложениями
        6. Включи цифры и факты из анализа
        
        Пример структуры:
        "На основе вашего запроса о [тема] мы обнаружили, что [ключевой инсайт]. 
        Это проявляется в [пример]. Мы рекомендуем [действие]."
        """
        
        return self.llm.generate(prompt)

    def _create_visualization(self, df: pd.DataFrame, vis_type: str, analysis: dict) -> dict:
        """Создает визуализацию данных"""
        if df.empty:
            return {"type": "text", "content": "Нет данных для визуализации"}
        
        try:
            if vis_type == "line_chart":
                return self._create_line_chart(df, analysis)
            elif vis_type == "bar_chart":
                return self._create_bar_chart(df, analysis)
            elif vis_type == "pie_chart":
                return self._create_pie_chart(df, analysis)
            elif vis_type == "scatter_plot":
                return self._create_scatter_plot(df, analysis)
            elif vis_type == "histogram":
                return self._create_histogram(df, analysis)
            elif vis_type == "summary_table":
                return self._create_summary_table(df, analysis)
            else:
                return self._create_bar_chart(df, analysis)  # Фолбэк
        except Exception as e:
            self.logger.warning(f"Visualization failed: {str(e)}")
            return self._create_summary_table(df, analysis)

    def _create_line_chart(self, df: pd.DataFrame, analysis: dict) -> dict:
        """Создает линейный график для временных рядов"""
        # Ищем колонку с датой
        date_col = next((col for col in df.columns if "date" in col.lower()), None)
        if not date_col:
            date_col = df.index.name if df.index.name else "index"
        
        # Ищем числовую колонку
        value_col = next((col for col in df.columns if df[col].dtype in ['int64', 'float64']), None)
        if not value_col:
            return self._create_summary_table(df, analysis)
        
        plt.figure(figsize=(10, 6))
        
        # Если есть категории для группировки
        category_col = next((col for col in df.columns if col not in [date_col, value_col] and df[col].nunique() < 10), None)
        
        if category_col:
            for i, (name, group) in enumerate(df.groupby(category_col)):
                plt.plot(group[date_col], group[value_col], 
                         label=name, color=self.color_palette[i], marker='o')
            plt.legend(title=category_col)
        else:
            plt.plot(df[date_col], df[value_col], color=self.color_palette[0], marker='o')
        
        plt.title(analysis.get("title", f'Динамика показателя "{value_col}"'))
        plt.xlabel(date_col)
        plt.ylabel(value_col)
        plt.grid(True)
        plt.xticks(rotation=45)
        
        return self._plot_to_base64(plt)

    def _create_bar_chart(self, df: pd.DataFrame, analysis: dict) -> dict:
        """Создает столбчатую диаграмму"""
        # Ищем категориальную колонку
        category_col = next((col for col in df.columns if df[col].nunique() < 20), None)
        if not category_col:
            category_col = df.index.name if df.index.name else "index"
        
        # Ищем числовую колонку
        value_col = next((col for col in df.columns if df[col].dtype in ['int64', 'float64']), None)
        if not value_col:
            # Если нет числовой колонки, считаем частоты
            value_counts = df[category_col].value_counts().reset_index()
            category_col = "index"
            value_col = value_counts.columns[1]
            df = value_counts
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=category_col, y=value_col, data=df, palette=self.color_palette)
        
        plt.title(analysis.get("title", f'Сравнение по "{category_col}"'))
        plt.xlabel(category_col)
        plt.ylabel(value_col)
        plt.xticks(rotation=45)
        
        return self._plot_to_base64(plt)

    def _create_pie_chart(self, df: pd.DataFrame, analysis: dict) -> dict:
        """Создает круговую диаграмму"""
        # Ищем категориальную колонку
        category_col = next((col for col in df.columns if df[col].nunique() < 10), None)
        if not category_col:
            return self._create_bar_chart(df, analysis)
        
        counts = df[category_col].value_counts()
        
        plt.figure(figsize=(8, 8))
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', 
                colors=self.color_palette, startangle=90)
        plt.title(analysis.get("title", f'Распределение по "{category_col}"'))
        
        return self._plot_to_base64(plt)

    def _create_scatter_plot(self, df: pd.DataFrame, analysis: dict) -> dict:
        """Создает диаграмму рассеяния"""
        num_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        if len(num_cols) < 2:
            return self._create_summary_table(df, analysis)
        
        x_col, y_col = num_cols[:2]
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x_col, y=y_col, data=df, 
                        hue=df[num_cols[2]] if len(num_cols) > 2 else None,
                        palette=self.color_palette)
        
        plt.title(analysis.get("title", f'Соотношение "{x_col}" и "{y_col}"'))
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        
        return self._plot_to_base64(plt)

    def _create_histogram(self, df: pd.DataFrame, analysis: dict) -> dict:
        """Создает гистограмму распределения"""
        num_col = next((col for col in df.columns if df[col].dtype in ['int64', 'float64']), None)
        if not num_col:
            return self._create_summary_table(df, analysis)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(df[num_col], kde=True, color=self.color_palette[0])
        
        plt.title(analysis.get("title", f'Распределение "{num_col}"'))
        plt.xlabel(num_col)
        plt.ylabel("Частота")
        
        return self._plot_to_base64(plt)

    def _create_summary_table(self, df: pd.DataFrame, analysis: dict) -> dict:
        """Создает текстовое представление таблицы"""
        # Берем только первые 5 строк и важные колонки
        display_df = df.head(5)
        if len(df.columns) > 5:
            display_df = display_df.iloc[:, :5]
        
        # Форматируем как Markdown таблицу
        table_md = display_df.to_markdown(index=False)
        
        # Добавляем заголовок
        title = analysis.get("title", "Обзор данных")
        return {
            "type": "text",
            "content": f"**{title}**\n\n```\n{table_md}\n```",
            "format": "markdown"
        }

    def _plot_to_base64(self, plt) -> dict:
        """Конвертирует matplotlib plot в base64"""
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return {
            "type": "image",
            "format": "base64",
            "data": img_base64
        }