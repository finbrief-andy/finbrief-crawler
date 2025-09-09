"""
Push Notification Service for Mobile Apps
Handles sending push notifications to iOS and Android devices.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import asyncio
import aiohttp
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class NotificationPriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class NotificationType(Enum):
    NEWS_ALERT = "news_alert"
    STRATEGY_UPDATE = "strategy_update"
    PRICE_ALERT = "price_alert"
    SYSTEM_MESSAGE = "system_message"


class PushNotification(BaseModel):
    """Push notification data model"""
    title: str
    body: str
    data: Optional[Dict[str, Any]] = None
    priority: NotificationPriority = NotificationPriority.NORMAL
    notification_type: NotificationType = NotificationType.SYSTEM_MESSAGE
    badge_count: Optional[int] = None
    sound: Optional[str] = "default"


class DeviceToken(BaseModel):
    """Device token data model"""
    token: str
    platform: str  # "ios" or "android"
    user_id: int
    app_version: str
    registered_at: datetime
    last_used: Optional[datetime] = None
    active: bool = True


class PushNotificationService:
    """Service for sending push notifications to mobile devices"""
    
    def __init__(self):
        self.firebase_server_key = os.getenv("FIREBASE_SERVER_KEY")
        self.apns_key_id = os.getenv("APNS_KEY_ID")
        self.apns_team_id = os.getenv("APNS_TEAM_ID")
        self.apns_bundle_id = os.getenv("APNS_BUNDLE_ID", "com.finbrief.app")
        
        # URLs for push services
        self.fcm_url = "https://fcm.googleapis.com/fcm/send"
        self.apns_url = "https://api.push.apple.com/3/device/"  # Production
        self.apns_dev_url = "https://api.development.push.apple.com/3/device/"  # Development
        
        self.logger = logging.getLogger(__name__)
        
        # In-memory device storage (in production, use database)
        self.device_tokens: List[DeviceToken] = []
    
    def register_device(self, token: str, platform: str, user_id: int, app_version: str = "1.0.0") -> bool:
        """Register a device token for push notifications"""
        try:
            # Remove existing tokens for this user/platform combination
            self.device_tokens = [
                dt for dt in self.device_tokens 
                if not (dt.user_id == user_id and dt.platform == platform)
            ]
            
            # Add new device token
            device_token = DeviceToken(
                token=token,
                platform=platform.lower(),
                user_id=user_id,
                app_version=app_version,
                registered_at=datetime.utcnow(),
                active=True
            )
            
            self.device_tokens.append(device_token)
            self.logger.info(f"Registered {platform} device for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering device: {e}")
            return False
    
    def get_user_devices(self, user_id: int) -> List[DeviceToken]:
        """Get all active devices for a user"""
        return [dt for dt in self.device_tokens if dt.user_id == user_id and dt.active]
    
    async def send_to_user(self, user_id: int, notification: PushNotification) -> Dict[str, Any]:
        """Send notification to all devices of a specific user"""
        devices = self.get_user_devices(user_id)
        
        if not devices:
            self.logger.warning(f"No devices found for user {user_id}")
            return {"success": False, "reason": "no_devices", "sent": 0}
        
        results = []
        for device in devices:
            if device.platform == "android":
                result = await self._send_fcm(device.token, notification)
            elif device.platform == "ios":
                result = await self._send_apns(device.token, notification)
            else:
                result = {"success": False, "error": "unknown_platform"}
            
            results.append({"device": device.token[:8], "result": result})
        
        successful_sends = sum(1 for r in results if r["result"].get("success"))
        
        return {
            "success": successful_sends > 0,
            "sent": successful_sends,
            "total_devices": len(devices),
            "results": results
        }
    
    async def send_to_multiple_users(self, user_ids: List[int], notification: PushNotification) -> Dict[str, Any]:
        """Send notification to multiple users"""
        tasks = []
        for user_id in user_ids:
            task = self.send_to_user(user_id, notification)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_sent = 0
        total_users = len(user_ids)
        
        for result in results:
            if isinstance(result, dict) and result.get("success"):
                total_sent += result.get("sent", 0)
        
        return {
            "success": total_sent > 0,
            "total_sent": total_sent,
            "total_users": total_users,
            "details": results
        }
    
    async def _send_fcm(self, token: str, notification: PushNotification) -> Dict[str, Any]:
        """Send push notification via Firebase Cloud Messaging (Android)"""
        if not self.firebase_server_key:
            return {"success": False, "error": "firebase_key_not_configured"}
        
        headers = {
            "Authorization": f"key={self.firebase_server_key}",
            "Content-Type": "application/json"
        }
        
        # Build FCM payload
        payload = {
            "to": token,
            "notification": {
                "title": notification.title,
                "body": notification.body,
                "sound": notification.sound,
                "icon": "ic_notification",
                "color": "#4285F4"  # FinBrief brand color
            },
            "data": notification.data or {},
            "priority": "high" if notification.priority == NotificationPriority.HIGH else "normal"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.fcm_url, headers=headers, json=payload) as response:
                    response_data = await response.json()
                    
                    if response.status == 200 and response_data.get("success", 0) > 0:
                        self.logger.info(f"FCM notification sent successfully to {token[:8]}")
                        return {"success": True, "response": response_data}
                    else:
                        self.logger.error(f"FCM notification failed: {response_data}")
                        return {"success": False, "error": response_data}
                        
        except Exception as e:
            self.logger.error(f"Error sending FCM notification: {e}")
            return {"success": False, "error": str(e)}
    
    async def _send_apns(self, token: str, notification: PushNotification) -> Dict[str, Any]:
        """Send push notification via Apple Push Notification Service (iOS)"""
        # For now, return success (APNS requires certificates/JWT tokens)
        # In production, implement proper APNS with certificates or JWT
        self.logger.info(f"APNS notification would be sent to {token[:8]} (mock)")
        
        # Mock APNS payload structure
        apns_payload = {
            "aps": {
                "alert": {
                    "title": notification.title,
                    "body": notification.body
                },
                "sound": notification.sound,
                "badge": notification.badge_count,
                "category": notification.notification_type.value
            },
            "data": notification.data or {}
        }
        
        return {
            "success": True, 
            "mock": True, 
            "payload": apns_payload,
            "message": "APNS mock - would send in production"
        }
    
    async def send_news_alert(self, user_ids: List[int], headline: str, summary: str, news_id: int):
        """Send news alert notification"""
        notification = PushNotification(
            title="📰 Breaking News",
            body=f"{headline[:80]}...",
            data={
                "type": "news",
                "news_id": news_id,
                "action": "open_news"
            },
            notification_type=NotificationType.NEWS_ALERT,
            priority=NotificationPriority.HIGH
        )
        
        return await self.send_to_multiple_users(user_ids, notification)
    
    async def send_strategy_update(self, user_ids: List[int], strategy_title: str, strategy_id: int, horizon: str):
        """Send strategy update notification"""
        notification = PushNotification(
            title="📊 Strategy Update",
            body=f"New {horizon} strategy: {strategy_title[:50]}...",
            data={
                "type": "strategy",
                "strategy_id": strategy_id,
                "horizon": horizon,
                "action": "open_strategy"
            },
            notification_type=NotificationType.STRATEGY_UPDATE,
            priority=NotificationPriority.NORMAL
        )
        
        return await self.send_to_multiple_users(user_ids, notification)
    
    async def send_price_alert(self, user_ids: List[int], symbol: str, price: float, change_percent: float):
        """Send price alert notification"""
        direction = "📈" if change_percent > 0 else "📉"
        notification = PushNotification(
            title=f"{direction} Price Alert",
            body=f"{symbol}: ${price:.2f} ({change_percent:+.1f}%)",
            data={
                "type": "price_alert",
                "symbol": symbol,
                "price": price,
                "change_percent": change_percent,
                "action": "open_ticker"
            },
            notification_type=NotificationType.PRICE_ALERT,
            priority=NotificationPriority.HIGH
        )
        
        return await self.send_to_multiple_users(user_ids, notification)
    
    def cleanup_inactive_tokens(self, days_threshold: int = 30):
        """Remove inactive device tokens"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
        
        before_count = len(self.device_tokens)
        self.device_tokens = [
            dt for dt in self.device_tokens 
            if dt.last_used and dt.last_used > cutoff_date
        ]
        
        removed_count = before_count - len(self.device_tokens)
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} inactive device tokens")
        
        return removed_count


# Global push notification service instance
_push_service = None

def get_push_service() -> PushNotificationService:
    """Get singleton push notification service instance"""
    global _push_service
    if _push_service is None:
        _push_service = PushNotificationService()
    return _push_service


# Utility functions for common notification scenarios
async def notify_breaking_news(user_ids: List[int], headline: str, news_id: int):
    """Quick function to send breaking news notifications"""
    service = get_push_service()
    return await service.send_news_alert(user_ids, headline, "", news_id)


async def notify_new_strategy(user_ids: List[int], strategy_title: str, strategy_id: int, horizon: str = "daily"):
    """Quick function to send new strategy notifications"""
    service = get_push_service()
    return await service.send_strategy_update(user_ids, strategy_title, strategy_id, horizon)


if __name__ == "__main__":
    # Test push notification service
    async def test_push_service():
        service = PushNotificationService()
        
        # Register test device
        service.register_device("test_token_123", "android", 1, "1.0.0")
        
        # Send test notification
        notification = PushNotification(
            title="Test Notification",
            body="This is a test notification from FinBrief",
            data={"test": True}
        )
        
        result = await service.send_to_user(1, notification)
        print(f"Test notification result: {result}")
    
    asyncio.run(test_push_service())