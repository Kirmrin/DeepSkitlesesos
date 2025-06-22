import logging
import psutil
import requests
import time
import json
import threading
import socket
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

class MonitoringService:
    def __init__(self):
        self.logger = logging.getLogger("monitoring_service")
        self.metrics = {}
        self.health_status = {}
        self.alert_rules = self._load_alert_rules()
        self.last_report_time = datetime.now()
        self.silenced_alerts = {}
        self.system_start_time = datetime.now()
        
    def start(self):
        """Запускает фоновые задачи мониторинга"""
        self.logger.info("Starting monitoring service")
        
        # Запуск фоновых задач
        threading.Thread(target=self._system_metrics_collector, daemon=True).start()
        threading.Thread(target=self._alert_checker, daemon=True).start()
        threading.Thread(target=self._report_generator, daemon=True).start()
        
    def record_metric(self, component: str, name: str, value: float, tags: Dict = None):
        """
        Регистрирует метрику производительности
        :param component: Название компонента (agent, service)
        :param name: Название метрики (response_time, error_rate)
        :param value: Значение метрики
        :param tags: Дополнительные теги ({"endpoint": "/query", "status": "200"})
        """
        metric_key = f"{component}.{name}"
        if metric_key not in self.metrics:
            self.metrics[metric_key] = {
                "values": [],
                "timestamps": [],
                "tags": tags or {}
            }
            
        self.metrics[metric_key]["values"].append(value)
        self.metrics[metric_key]["timestamps"].append(datetime.now())
        
        # Сохраняем только последние 100 значений
        if len(self.metrics[metric_key]["values"]) > 100:
            self.metrics[metric_key]["values"].pop(0)
            self.metrics[metric_key]["timestamps"].pop(0)
    
    def log_error(self, component: str, error_type: str, message: str, details: Dict = None):
        """
        Регистрирует ошибку
        :param component: Компонент системы
        :param error_type: Тип ошибки (critical, warning, info)
        :param message: Описание ошибки
        :param details: Дополнительные детали
        """
        error_key = f"{component}.{error_type}"
        if error_key not in self.metrics:
            self.metrics[error_key] = {
                "count": 0,
                "last_occurrence": None,
                "messages": []
            }
            
        self.metrics[error_key]["count"] += 1
        self.metrics[error_key]["last_occurrence"] = datetime.now()
        self.metrics[error_key]["messages"].append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "details": details or {}
        })
        
        # Сохраняем только последние 20 сообщений
        if len(self.metrics[error_key]["messages"]) > 20:
            self.metrics[error_key]["messages"].pop(0)
            
        # Обновляем статус здоровья компонента
        if error_type == "critical":
            self.health_status[component] = "critical"
        elif error_type == "warning" and self.health_status.get(component) != "critical":
            self.health_status[component] = "warning"
            
        # Отправка алерта
        self._evaluate_alert_rules(component, error_type, message)
    
    def register_health_check(self, component: str, check_func: callable):
        """
        Регистрирует функцию проверки здоровья компонента
        :param component: Название компонента
        :param check_func: Функция, возвращающая (status: str, details: dict)
        """
        self.health_status[component] = {
            "check_func": check_func,
            "last_check": None,
            "last_status": "unknown",
            "last_details": {}
        }
    
    def get_system_health(self) -> Dict[str, str]:
        """Возвращает текущий статус здоровья компонентов"""
        return {comp: data["last_status"] for comp, data in self.health_status.items()}
    
    def get_metrics(self, component: str = None, name: str = None) -> Dict:
        """Возвращает метрики по фильтру"""
        if component and name:
            return self.metrics.get(f"{component}.{name}", {})
        elif component:
            return {k: v for k, v in self.metrics.items() if k.startswith(f"{component}.")}
        return self.metrics
    
    def silence_alert(self, alert_id: str, duration_minutes: int = 60):
        """
        Отключает алерт на указанное время
        :param alert_id: ID алерта
        :param duration_minutes: Длительность в минутах
        """
        self.silenced_alerts[alert_id] = datetime.now() + timedelta(minutes=duration_minutes)
        self.logger.info(f"Alert {alert_id} silenced for {duration_minutes} minutes")
    
    def _system_metrics_collector(self):
        """Фоновая задача для сбора системных метрик"""
        while True:
            try:
                # CPU
                cpu_percent = psutil.cpu_percent(interval=1)
                self.record_metric("system", "cpu_usage", cpu_percent)
                
                # Memory
                memory = psutil.virtual_memory()
                self.record_metric("system", "memory_usage", memory.percent)
                self.record_metric("system", "memory_used_gb", memory.used / (1024**3))
                
                # Disk
                disk = psutil.disk_usage('/')
                self.record_metric("system", "disk_usage", disk.percent)
                
                # Network
                net_io = psutil.net_io_counters()
                self.record_metric("system", "network_sent_mb", net_io.bytes_sent / (1024**2))
                self.record_metric("system", "network_recv_mb", net_io.bytes_recv / (1024**2))
                
                # Process count
                process_count = len(psutil.pids())
                self.record_metric("system", "process_count", process_count)
                
                # Uptime
                uptime = (datetime.now() - self.system_start_time).total_seconds()
                self.record_metric("system", "uptime_seconds", uptime)
                
            except Exception as e:
                self.logger.error(f"System metrics collection failed: {str(e)}")
                
            time.sleep(60)
    
    def _alert_checker(self):
        """Фоновая задача для проверки алерт-правил"""
        while True:
            # Проверка здоровья компонентов
            for component, data in self.health_status.items():
                if callable(data.get("check_func")):
                    try:
                        status, details = data["check_func"]()
                        data["last_status"] = status
                        data["last_details"] = details
                        data["last_check"] = datetime.now()
                        
                        if status == "critical":
                            self._trigger_alert(
                                f"{component}_health_critical",
                                f"Component {component} is in critical state",
                                details
                            )
                        elif status == "warning":
                            self._trigger_alert(
                                f"{component}_health_warning",
                                f"Component {component} is in warning state",
                                details
                            )
                            
                    except Exception as e:
                        self.logger.error(f"Health check for {component} failed: {str(e)}")
            
            # Проверка метрик по правилам
            for rule in self.alert_rules:
                if self._should_check_rule(rule):
                    value = self._get_metric_value(rule["metric"])
                    if value is not None and self._evaluate_condition(value, rule["condition"], rule["threshold"]):
                        self._trigger_alert(
                            rule["id"],
                            rule["message"],
                            {
                                "metric": rule["metric"],
                                "value": value,
                                "threshold": rule["threshold"]
                            }
                        )
            
            time.sleep(30)
    
    def _report_generator(self):
        """Генерация периодических отчетов"""
        while True:
            try:
                now = datetime.now()
                if now - self.last_report_time > timedelta(hours=1):
                    self._generate_hourly_report()
                    self.last_report_time = now
                    
                time.sleep(60)
            except Exception as e:
                self.logger.error(f"Report generation failed: {str(e)}")
    
    def _trigger_alert(self, alert_id: str, message: str, details: Dict):
        """Активирует алерт, если он не заглушен"""
        # Проверка заглушки
        if alert_id in self.silenced_alerts and datetime.now() < self.silenced_alerts[alert_id]:
            return
            
        # Отправка алерта
        self._send_alert(alert_id, message, details)
        
        # Логирование
        self.logger.warning(f"ALERT: {alert_id} - {message}")
        self.log_error("monitoring", "alert", message, details)
        
        # Временная заглушка для предотвращения флуда
        self.silenced_alerts[alert_id] = datetime.now() + timedelta(minutes=5)
    
    def _send_alert(self, alert_id: str, message: str, details: Dict):
        """Отправляет алерт во внешние системы"""
        # Отправка в Telegram
        if config.TELEGRAM_ALERTS_ENABLED:
            try:
                text = f"🚨 *{alert_id}*\n{message}\n\n"
                text += "```json\n" + json.dumps(details, indent=2) + "\n```"
                
                requests.post(
                    f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
                    json={
                        "chat_id": config.TELEGRAM_ALERT_CHAT_ID,
                        "text": text,
                        "parse_mode": "Markdown"
                    }
                )
            except Exception as e:
                self.logger.error(f"Failed to send Telegram alert: {str(e)}")
        
        # Отправка в Slack
        if config.SLACK_ALERTS_ENABLED:
            try:
                requests.post(
                    config.SLACK_WEBHOOK_URL,
                    json={
                        "text": f":fire: {alert_id}",
                        "blocks": [
                            {
                                "type": "section",
                                "text": {"type": "mrkdwn", "text": f"*{alert_id}*\n{message}"}
                            },
                            {
                                "type": "section",
                                "text": {"type": "mrkdwn", "text": f"```{json.dumps(details, indent=2)}```"}
                            }
                        ]
                    }
                )
            except Exception as e:
                self.logger.error(f"Failed to send Slack alert: {str(e)}")
        
        # Отправка в Email (заглушка)
        if config.EMAIL_ALERTS_ENABLED:
            self.logger.info(f"Email alert would be sent: {alert_id} - {message}")
    
    def _generate_hourly_report(self):
        """Генерирует и отправляет часовой отчет"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "uptime": str(datetime.now() - self.system_start_time),
            "health_status": self.get_system_health(),
            "top_metrics": {}
        }
        
        # Сбор топ-метрик
        for metric, data in self.metrics.items():
            if data.get("values"):
                report["top_metrics"][metric] = {
                    "last": data["values"][-1],
                    "avg": sum(data["values"]) / len(data["values"]),
                    "min": min(data["values"]),
                    "max": max(data["values"])
                }
        
        # Отправка отчета
        if config.REPORTING_ENABLED:
            try:
                # Сохранение в файл
                with open(f"reports/system_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json", "w") as f:
                    json.dump(report, f, indent=2)
                
                # Отправка в Telegram (если настроено)
                if config.TELEGRAM_REPORTS_ENABLED:
                    text = f"📊 *System Report*\nUptime: {report['uptime']}\n"
                    text += f"Health: {', '.join([f'{k}:{v}' for k, v in report['health_status'].items()])}"
                    
                    requests.post(
                        f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
                        json={
                            "chat_id": config.TELEGRAM_REPORT_CHAT_ID,
                            "text": text,
                            "parse_mode": "Markdown"
                        }
                    )
                    
            except Exception as e:
                self.logger.error(f"Failed to generate report: {str(e)}")
    
    def _load_alert_rules(self) -> List[Dict]:
        """Загружает правила генерации алертов"""
        return [
            {
                "id": "high_cpu",
                "metric": "system.cpu_usage",
                "condition": ">",
                "threshold": 90,
                "message": "High CPU usage detected",
                "interval": 300
            },
            {
                "id": "high_memory",
                "metric": "system.memory_usage",
                "condition": ">",
                "threshold": 90,
                "message": "High memory usage detected",
                "interval": 300
            },
            {
                "id": "high_error_rate",
                "metric": "system.error_rate",
                "condition": ">",
                "threshold": 10,
                "message": "High error rate detected",
                "interval": 600
            },
            {
                "id": "component_down",
                "metric": "health_status",
                "condition": "==",
                "threshold": "critical",
                "message": "Critical component failure",
                "interval": 60
            }
        ]
    
    def _should_check_rule(self, rule: Dict) -> bool:
        """Определяет, нужно ли проверять правило сейчас"""
        last_trigger = self.metrics.get(f"alerts.{rule['id']}", {}).get("last_trigger")
        if last_trigger and (datetime.now() - last_trigger).total_seconds() < rule["interval"]:
            return False
        return True
    
    def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Возвращает последнее значение метрики"""
        if metric_name == "health_status":
            # Специальная обработка для статуса здоровья
            return "ok" if all(s == "ok" for s in self.get_system_health().values()) else "warning"
        
        metric_data = self.metrics.get(metric_name)
        if metric_data and "values" in metric_data and metric_data["values"]:
            return metric_data["values"][-1]
        return None
    
    def _evaluate_condition(self, value, condition: str, threshold) -> bool:
        """Проверяет условие алерта"""
        try:
            if condition == ">":
                return value > threshold
            elif condition == ">=":
                return value >= threshold
            elif condition == "<":
                return value < threshold
            elif condition == "<=":
                return value <= threshold
            elif condition == "==":
                return value == threshold
            elif condition == "!=":
                return value != threshold
            return False
        except TypeError:
            return False
    
    def add_custom_alert_rule(self, rule: Dict):
        """Добавляет пользовательское правило алертинга"""
        self.alert_rules.append(rule)
        self.logger.info(f"Added custom alert rule: {rule['id']}")
    
    def get_service_info(self) -> Dict:
        """Возвращает информацию о сервисе для health-check"""
        return {
            "status": "running",
            "start_time": self.system_start_time.isoformat(),
            "components_monitored": list(self.health_status.keys()),
            "metrics_collected": list(self.metrics.keys())
        }

# Пример функции health-check для других компонентов
def sql_generator_health_check() -> Tuple[str, Dict]:
    """Пример функции проверки здоровья для SQLGenerator"""
    try:
        # Проверка подключения к LLM
        # Проверка доступности схемы БД
        return "ok", {"details": "All dependencies available"}
    except Exception as e:
        return "critical", {"error": str(e)}