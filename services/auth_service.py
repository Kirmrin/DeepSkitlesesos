import logging
import config
import jwt
import time
import hashlib
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

class AuthService:
    def __init__(self):
        self.logger = logging.getLogger("auth_service")
        self.user_store = self._load_user_store()
        self.roles = self._load_roles_config()
        self.secret_key = config.SECRET_KEY
        self.token_cache = {}
        
    def authenticate_user(self, telegram_data: Dict) -> Tuple[bool, Dict]:
        """
        Аутентификация пользователя Telegram
        :param telegram_data: Данные от Telegram WebApp (id, first_name, auth_date, hash)
        :return: (success: bool, user_data: dict)
        """
        try:
            # Проверка подписи данных
            if not self._verify_telegram_data(telegram_data):
                return False, {"error": "invalid_signature"}
            
            user_id = telegram_data["id"]
            
            # Поиск существующего пользователя
            if user_id in self.user_store:
                user = self.user_store[user_id]
                user["last_login"] = datetime.now().isoformat()
                return True, user
            
            # Создание нового пользователя
            new_user = self._create_new_user(telegram_data)
            self.user_store[user_id] = new_user
            self._save_user_store()
            
            return True, new_user
            
        except Exception as e:
            self.logger.error(f"Authentication failed: {str(e)}")
            return False, {"error": "authentication_error"}

    def _verify_telegram_data(self, data: Dict) -> bool:
        """Проверка подписи данных от Telegram"""
        # Алгоритм проверки согласно документации Telegram
        received_hash = data.pop("hash")
        data_check_string = "\n".join(
            f"{key}={value}" for key, value in sorted(data.items()))
        
        secret_key = hashlib.sha256(config.TELEGRAM_BOT_TOKEN.encode()).digest()
        computed_hash = hashlib.hmac256(
            secret_key, 
            data_check_string.encode()
        ).hexdigest()
        
        return computed_hash == received_hash

    def _create_new_user(self, telegram_data: Dict) -> Dict:
        """Создание профиля нового пользователя"""
        default_role = "user"
        
        return {
            "id": telegram_data["id"],
            "username": telegram_data.get("username", f"user_{telegram_data['id']}"),
            "first_name": telegram_data.get("first_name", ""),
            "last_name": telegram_data.get("last_name", ""),
            "language": telegram_data.get("language_code", "ru"),
            "role": default_role,
            "created_at": datetime.now().isoformat(),
            "last_login": datetime.now().isoformat(),
            "permissions": self.roles.get(default_role, {}).get("permissions", [])
        }

    def authorize(self, user_id: int, permission: str) -> bool:
        """
        Проверка прав доступа пользователя
        :param user_id: ID пользователя
        :param permission: Требуемое разрешение ("analytics:read", "export:create" и т.д.)
        :return: bool
        """
        user = self.user_store.get(user_id)
        if not user:
            return False
            
        # Админы имеют все права
        if user["role"] == "admin":
            return True
            
        # Проверка конкретного разрешения
        return permission in user.get("permissions", [])

    def generate_jwt(self, user_id: int) -> str:
        """Генерация JWT токена для пользователя"""
        # Проверка кеша
        if user_id in self.token_cache:
            token, exp = self.token_cache[user_id]
            if time.time() < exp - 60:  # Если токен почти не истек
                return token
        
        # Создание нового токена
        user = self.user_store[user_id]
        payload = {
            "sub": user_id,
            "role": user["role"],
            "exp": datetime.utcnow() + timedelta(hours=config.JWT_EXP_HOURS)
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        self.token_cache[user_id] = (token, payload["exp"])
        return token

    def verify_jwt(self, token: str) -> Optional[Dict]:
        """Верификация JWT токена"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token expired")
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid JWT token")
        return None

    def get_user_context(self, user_id: int) -> Dict:
        """Возвращает контекст пользователя для системы"""
        user = self.user_store.get(user_id)
        if not user:
            return {}
            
        return {
            "user_id": user_id,
            "role": user["role"],
            "language": user["language"],
            "allowed_tables": self._get_allowed_tables(user["role"]),
            "permissions": user["permissions"]
        }

    def _get_allowed_tables(self, role: str) -> List[str]:
        """Возвращает список доступных таблиц для роли"""
        role_config = self.roles.get(role, {})
        return role_config.get("allowed_tables", ["*"])

    def update_user_role(self, user_id: int, new_role: str) -> bool:
        """Обновление роли пользователя (только для админов)"""
        if new_role not in self.roles:
            return False
            
        user = self.user_store.get(user_id)
        if not user:
            return False
            
        user["role"] = new_role
        user["permissions"] = self.roles[new_role].get("permissions", [])
        self._save_user_store()
        return True

    def _load_user_store(self) -> Dict:
        """Загрузка хранилища пользователей (в реальной системе - из БД)"""
        # В демо-версии используем файл, в продакшене - подключение к БД
        try:
            with open(config.USER_STORE_FILE, "r") as f:
                return json.load(f)
        except:
            return {}

    def _save_user_store(self):
        """Сохранение хранилища пользователей"""
        try:
            with open(config.USER_STORE_FILE, "w") as f:
                json.dump(self.user_store, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save user store: {str(e)}")

    def _load_roles_config(self) -> Dict:
        """Загрузка конфигурации ролей"""
        return {
            "admin": {
                "permissions": ["*"],
                "allowed_tables": ["*"],
                "description": "Полный доступ"
            },
            "analyst": {
                "permissions": ["analytics:read", "export:create", "data:query"],
                "allowed_tables": ["sales", "customers", "products"],
                "description": "Аналитик данных"
            },
            "manager": {
                "permissions": ["analytics:read", "reports:view"],
                "allowed_tables": ["sales_summary", "kpi"],
                "description": "Менеджер"
            },
            "user": {
                "permissions": ["analytics:read"],
                "allowed_tables": ["public_data"],
                "description": "Обычный пользователь"
            }
        }