import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import tempfile
import os
import logging
import json
from typing import Dict, Any, List, Union
from config import Config
from services.monitoring import tracer
import numpy as np
import seaborn as sns
from datetime import datetime

# Настройка логгера
logger = logging.getLogger(__name__)

class ResponseFormatter:
    def __init__(self):
        # Настройки форматирования
        self.max_text_length = Config.MAX_TEXT_LENGTH or 4000  # Ограничение Telegram
        self.max_table_rows = Config.MAX_TABLE_ROWS or 20
        self.max_table_columns = Config.MAX_TABLE_COLUMNS or 8
        self.image_quality = Config.IMAGE_QUALITY or 90
        self.temp_dir = Config.TEMP_DIR or tempfile.gettempdir()
        
        # Поддерживаемые форматы
        self.supported_formats = ["text", "table", "image", "file", "markdown"]
    
    def format(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Форматирует ответ для отправки через Telegram API
        
        :param response_data: Данные ответа от агентов
        :return: Словарь в формате, готовом для отправки в Telegram
        """
        try:
            response_type = response_data.get("type", "text")
            
            with tracer.trace("response_formatting") as span:
                span.input = json.dumps(response_data, default=str)
                
                if response_type == "text":
                    result = self._format_text(response_data)
                elif response_type == "table":
                    result = self._format_table(response_data)
                elif response_type == "image":
                    result = self._format_image(response_data)
                elif response_type == "file":
                    result = self._format_file(response_data)
                elif response_type == "markdown":
                    result = self._format_markdown(response_data)
                else:
                    logger.warning(f"Неизвестный тип ответа: {response_type}")
                    result = self._format_text({
                        "content": "Неподдерживаемый формат ответа",
                        "sources": response_data.get("sources", [])
                    })
                
                span.output = json.dumps(result, ensure_ascii=False)
                return result
        
        except Exception as e:
            logger.error(f"Ошибка форматирования ответа: {str(e)}")
            return self._format_error(str(e))

    def _format_text(self, data: Dict) -> Dict[str, Any]:
        """Форматирует текстовый ответ"""
        content = data.get("content", "")
        sources = data.get("sources", [])
        
        # Добавляем источники, если есть
        if sources:
            sources_text = "\n\nИсточники:\n" + "\n".join(f"• {src}" for src in sources[:3])
            content += sources_text
        
        # Обрезаем слишком длинный текст
        if len(content) > self.max_text_length:
            content = content[:self.max_text_length - 100] + "...\n\n[сообщение сокращено]"
        
        return {
            "type": "text",
            "content": content,
            "parse_mode": "HTML"
        }

    def _format_table(self, data: Dict) -> Dict[str, Any]:
        """Форматирует табличные данные"""
        table_data = data.get("data")
        
        if isinstance(table_data, pd.DataFrame):
            df = table_data
        elif isinstance(table_data, list):
            df = pd.DataFrame(table_data)
        else:
            return self._format_error("Некорректные данные таблицы")
        
        # Ограничение размера таблицы
        if len(df) > self.max_table_rows or len(df.columns) > self.max_table_columns:
            return self._convert_to_image(df, data)
        
        # Форматирование в Markdown
        try:
            return self._format_as_markdown_table(df, data)
        except Exception as e:
            logger.warning(f"Ошибка форматирования таблицы: {str(e)}")
            return self._convert_to_image(df, data)

    def _format_as_markdown_table(self, df: pd.DataFrame, data: Dict) -> Dict[str, Any]:
        """Конвертирует DataFrame в таблицу Markdown"""
        # Ограничиваем количество строк и столбцов
        df = df.head(self.max_table_rows)
        if len(df.columns) > self.max_table_columns:
            df = df[df.columns[:self.max_table_columns]]
        
        # Создаем Markdown таблицу
        headers = df.columns.tolist()
        md_table = "| " + " | ".join(headers) + " |\n"
        md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        
        for _, row in df.iterrows():
            values = [str(row[col])[:50] for col in headers]  # Обрезаем длинные значения
            md_table += "| " + " | ".join(values) + " |\n"
        
        # Добавляем заголовок
        title = data.get("title", "Результаты запроса")
        content = f"<b>{title}</b>\n\n```\n{md_table}\n```"
        
        # Добавляем источники
        if sources := data.get("sources"):
            content += "\n\nИсточники: " + ", ".join(sources[:2])
        
        return {
            "type": "text",
            "content": content,
            "parse_mode": "HTML"
        }

    def _convert_to_image(self, df: pd.DataFrame, data: Dict) -> Dict[str, Any]:
        """Конвертирует таблицу в изображение, если она слишком большая"""
        try:
            # Создаем фигуру
            plt.figure(figsize=(12, min(8, 0.3 * len(df))))
            
            # Создаем табличный график
            ax = plt.subplot(111, frame_on=False)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            
            # Ограничиваем данные для отображения
            display_df = df.head(50)
            
            # Создаем таблицу
            table = plt.table(
                cellText=display_df.values,
                colLabels=display_df.columns,
                cellLoc='center',
                loc='center'
            )
            
            # Настраиваем стиль
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)
            
            # Заголовок
            title = data.get("title", "Результаты запроса")
            plt.title(title, fontsize=12)
            
            # Сохраняем в BytesIO
            buf = io.BytesIO()
            plt.savefig(
                buf, 
                format='png', 
                bbox_inches='tight',
                dpi=120,
                quality=self.image_quality
            )
            buf.seek(0)
            
            # Кодируем в base64
            img_str = base64.b64encode(buf.read()).decode()
            plt.close()
            
            return {
                "type": "image",
                "content": title,
                "data": f"data:image/png;base64,{img_str}",
                "sources": data.get("sources", [])
            }
        
        except Exception as e:
            logger.error(f"Ошибка конвертации таблицы в изображение: {str(e)}")
            return self._format_text({
                "content": f"Не удалось отобразить таблицу. {str(e)}",
                "sources": data.get("sources", [])
            })

    def _format_image(self, data: Dict) -> Dict[str, Any]:
        """Форматирует изображение"""
        # Если изображение в base64
        if "base64" in data:
            return {
                "type": "image",
                "data": f"data:image/png;base64,{data['base64']}",
                "caption": data.get("caption", ""),
                "sources": data.get("sources", [])
            }
        
        # Если URL изображения
        elif "url" in data:
            return {
                "type": "image",
                "url": data["url"],
                "caption": data.get("caption", ""),
                "sources": data.get("sources", [])
            }
        
        # Если данные для построения графика
        elif "plot_data" in data:
            return self._generate_plot(data)
        
        return self._format_error("Некорректные данные изображения")

    def _generate_plot(self, data: Dict) -> Dict[str, Any]:
        """Генерирует график на основе данных"""
        try:
            plt.figure(figsize=(10, 6))
            
            plot_type = data.get("plot_type", "bar")
            plot_data = data["plot_data"]
            
            if plot_type == "bar":
                plt.bar(plot_data["x"], plot_data["y"])
            elif plot_type == "line":
                plt.plot(plot_data["x"], plot_data["y"])
            elif plot_type == "pie":
                plt.pie(plot_data["sizes"], labels=plot_data["labels"])
            elif plot_type == "scatter":
                plt.scatter(plot_data["x"], plot_data["y"])
            elif plot_type == "hist":
                plt.hist(plot_data["values"])
            else:
                # Используем Seaborn для сложных графиков
                if plot_type == "heatmap":
                    sns.heatmap(pd.DataFrame(plot_data["matrix"]), annot=True)
                elif plot_type == "boxplot":
                    sns.boxplot(data=pd.DataFrame(plot_data["data"]))
                else:
                    # По умолчанию линейный график
                    plt.plot(plot_data["x"], plot_data["y"])
            
            # Заголовки и подписи
            plt.title(data.get("title", "Результаты анализа"))
            plt.xlabel(data.get("x_label", ""))
            plt.ylabel(data.get("y_label", ""))
            
            if plot_type != "pie":
                plt.grid(True, linestyle='--', alpha=0.7)
            
            # Сохраняем в BytesIO
            buf = io.BytesIO()
            plt.savefig(
                buf, 
                format='png', 
                bbox_inches='tight',
                dpi=120,
                quality=self.image_quality
            )
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            plt.close()
            
            return {
                "type": "image",
                "content": data.get("title", "График результатов"),
                "data": f"data:image/png;base64,{img_str}",
                "sources": data.get("sources", [])
            }
        
        except Exception as e:
            logger.error(f"Ошибка генерации графика: {str(e)}")
            return self._format_text({
                "content": f"Не удалось сгенерировать график. {str(e)}",
                "sources": data.get("sources", [])
            })

    def _format_file(self, data: Dict) -> Dict[str, Any]:
        """Форматирует файловый ответ"""
        # Если CSV данные
        if "csv_data" in data:
            return self._create_csv_file(data)
        
        # Если Excel данные
        elif "excel_data" in data:
            return self._create_excel_file(data)
        
        # Если путь к файлу
        elif "file_path" in data and os.path.exists(data["file_path"]):
            return {
                "type": "file",
                "file_path": data["file_path"],
                "filename": data.get("filename", os.path.basename(data["file_path"])),
                "caption": data.get("caption", ""),
                "sources": data.get("sources", [])
            }
        
        return self._format_error("Некорректные данные файла")

    def _create_csv_file(self, data: Dict) -> Dict[str, Any]:
        """Создает временный CSV файл"""
        try:
            df = pd.DataFrame(data["csv_data"])
            
            # Создаем временный файл
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = data.get("filename", f"data_{timestamp}.csv")
            file_path = os.path.join(self.temp_dir, filename)
            
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            return {
                "type": "file",
                "file_path": file_path,
                "filename": filename,
                "caption": data.get("caption", "Данные в формате CSV"),
                "sources": data.get("sources", [])
            }
        except Exception as e:
            logger.error(f"Ошибка создания CSV: {str(e)}")
            return self._format_error(f"Ошибка создания файла: {str(e)}")

    def _create_excel_file(self, data: Dict) -> Dict[str, Any]:
        """Создает временный Excel файл"""
        try:
            # Данные могут быть несколькими листами
            excel_data = data["excel_data"]
            
            # Создаем временный файл
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = data.get("filename", f"data_{timestamp}.xlsx")
            file_path = os.path.join(self.temp_dir, filename)
            
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                for sheet_name, sheet_data in excel_data.items():
                    if isinstance(sheet_data, pd.DataFrame):
                        sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
                    else:
                        pd.DataFrame(sheet_data).to_excel(writer, sheet_name=sheet_name, index=False)
            
            return {
                "type": "file",
                "file_path": file_path,
                "filename": filename,
                "caption": data.get("caption", "Данные в формате Excel"),
                "sources": data.get("sources", [])
            }
        except Exception as e:
            logger.error(f"Ошибка создания Excel: {str(e)}")
            return self._format_error(f"Ошибка создания файла: {str(e)}")

    def _format_markdown(self, data: Dict) -> Dict[str, Any]:
        """Форматирует Markdown-контент"""
        content = data.get("content", "")
        sources = data.get("sources", [])
        
        # Добавляем источники, если есть
        if sources:
            content += "\n\n**Источники:**\n" + "\n".join(f"- {src}" for src in sources[:3])
        
        # Обрезаем слишком длинный текст
        if len(content) > self.max_text_length:
            content = content[:self.max_text_length - 100] + "...\n\n[сообщение сокращено]"
        
        return {
            "type": "text",
            "content": content,
            "parse_mode": "MarkdownV2"
        }

    def _format_error(self, message: str) -> Dict[str, Any]:
        """Форматирует сообщение об ошибке"""
        return {
            "type": "text",
            "content": f"⚠️ Ошибка форматирования ответа: {message}",
            "parse_mode": "HTML"
        }