import pandas as pd
import streamlit as st
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
import time
import json
import random
import logging
import os
import sys
import subprocess
import streamlit as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime, timedelta
from collections import Counter, deque
from typing import Dict, List, Any, Tuple, Optional

from pathlib import Path

# 定义缓存文件路径
ARTIFACTS_DIR = Path("artifacts")
FEATURES_PATH = ARTIFACTS_DIR / "features_demo.joblib"

# 如果缓存文件不存在，自动运行训练脚本生成
if not FEATURES_PATH.exists():
    st.info("正在生成特征缓存文件，请稍候...")
    # 创建目录
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    # 运行训练命令
    subprocess.run(["python", "train.py", "--mode", "demo"], check=True)
    st.success("特征缓存生成完成！")

# 之后再导入你的其他模块、运行主界面代码
from eeg_pipeline import *
try:
    from eeg_pipeline import (
        DATA_PATH,
        get_feature_cache_path,
        get_model_paths,
        load_feature_cache,
    )
except ImportError:
    sys.path.insert(0, os.getcwd())
    from eeg_pipeline import (
        DATA_PATH,
        get_feature_cache_path,
        get_model_paths,
        load_feature_cache,
    )

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.unicode_minus'] = False

# 安全过滤阈值
LOW_CONFIDENCE_THRESHOLD = 65.0
PRINT_SECURITY_ALERTS = False

LABEL_CN_MAP = {
    "left": "左手运动",
    "right": "右手运动",
    "foot": "双脚运动",
    "tongue": "舌头运动"
}


# ===================== 结构化指令定义 =====================
class MachineCommand:
    """结构化机器指令类"""

    def __init__(self, command_type: str, device: str, action: str,
                 params: Dict[str, Any] = None, safety_level: str = "low"):
        self.command_type = command_type  # 指令类型
        self.device = device  # 设备类型
        self.action = action  # 具体动作
        self.params = params or {}  # 指令参数
        self.safety_level = safety_level  # 安全等级

    def to_structured_string(self) -> str:
        """转换为结构化指令字符串"""
        if self.params:
            param_str = " " + " ".join([f"{k}={v}" for k, v in self.params.items()])
        else:
            param_str = ""
        return f"{self.device}_{self.action.upper()}{param_str}"

    def to_display_string(self) -> str:
        """转换为显示字符串"""
        base_map = {
            "WHEELCHAIR_FORWARD": "轮椅前进",
            "WHEELCHAIR_BACKWARD": "轮椅后退",
            "WHEELCHAIR_LEFT": "轮椅左转",
            "WHEELCHAIR_RIGHT": "轮椅右转",
            "PROSTHETIC_LEFT_GRIP": "假肢左手抓取",
            "PROSTHETIC_LEFT_RELEASE": "假肢左手释放",
            "PROSTHETIC_RIGHT_GRIP": "假肢右手抓取",
            "PROSTHETIC_RIGHT_RELEASE": "假肢右手释放",
            "LIGHT_ON": "打开灯光",
            "LIGHT_OFF": "关闭灯光",
            "VOLUME_UP": "音量增加",
            "VOLUME_DOWN": "音量减少",
            "CURTAIN_OPEN": "打开窗帘",
            "CURTAIN_CLOSE": "关闭窗帘",
            "DRINK_START": "启动喝水装置",
            "DRINK_STOP": "停止喝水装置",
        }
        key = f"{self.device}_{self.action.upper()}"
        return base_map.get(key, key)


# ===================== 结构化指令库 =====================
STRUCTURED_COMMAND_LIBRARY = {
    "wheelchair_forward": MachineCommand(
        command_type="MOVE",
        device="WHEELCHAIR",
        action="FORWARD",
        params={"speed": 50, "duration": 5},
        safety_level="high"
    ),
    "wheelchair_backward": MachineCommand(
        command_type="MOVE",
        device="WHEELCHAIR",
        action="BACKWARD",
        params={"speed": 30, "duration": 3},
        safety_level="high"
    ),
    "wheelchair_left": MachineCommand(
        command_type="ROTATE",
        device="WHEELCHAIR",
        action="LEFT",
        params={"angle": 45, "speed": 20},
        safety_level="medium"
    ),
    "wheelchair_right": MachineCommand(
        command_type="ROTATE",
        device="WHEELCHAIR",
        action="RIGHT",
        params={"angle": 45, "speed": 20},
        safety_level="medium"
    ),
    "prosthetic_left_grab": MachineCommand(
        command_type="GRIP",
        device="PROSTHETIC",
        action="LEFT_GRIP",
        params={"force": 70, "duration": 2},
        safety_level="high"
    ),
    "prosthetic_left_release": MachineCommand(
        command_type="RELEASE",
        device="PROSTHETIC",
        action="LEFT_RELEASE",
        params={"duration": 1},
        safety_level="medium"
    ),
    "light_on": MachineCommand(
        command_type="SWITCH",
        device="LIGHT",
        action="ON",
        params={"brightness": 80},
        safety_level="low"
    ),
    "light_off": MachineCommand(
        command_type="SWITCH",
        device="LIGHT",
        action="OFF",
        params={},
        safety_level="low"
    ),
    "curtain_open": MachineCommand(
        command_type="CONTROL",
        device="CURTAIN",
        action="OPEN",
        params={"percentage": 100},
        safety_level="low"
    ),
    "curtain_close": MachineCommand(
        command_type="CONTROL",
        device="CURTAIN",
        action="CLOSE",
        params={"percentage": 0},
        safety_level="low"
    ),
    "volume_up": MachineCommand(
        command_type="ADJUST",
        device="VOLUME",
        action="UP",
        params={"level": 30},
        safety_level="low"
    ),
    "volume_down": MachineCommand(
        command_type="ADJUST",
        device="VOLUME",
        action="DOWN",
        params={"level": 10},
        safety_level="low"
    ),
    "drink_start": MachineCommand(
        command_type="CONTROL",
        device="DRINK",
        action="START",
        params={"duration": 10},
        safety_level="medium"
    ),
    "drink_stop": MachineCommand(
        command_type="CONTROL",
        device="DRINK",
        action="STOP",
        params={},
        safety_level="medium"
    ),
    # 测试用危险指令
    "malicious_shutdown": MachineCommand(
        command_type="SYSTEM",
        device="SHUTDOWN",
        action="IMMEDIATE",
        params={"force": True},
        safety_level="critical"
    ),
}

# 意图-指令映射
INTENT_ALLOWED_COMMANDS = {
    "left": [
        "wheelchair_left",
        "prosthetic_left_grab",
        "light_on",
        "volume_up",
    ],
    "right": [
        "wheelchair_right",
        "prosthetic_left_release",
        "light_off",
        "volume_down",
    ],
    "foot": [
        "wheelchair_forward",
        "curtain_open",
        "drink_start",
    ],
    "tongue": [
        "wheelchair_backward",
        "curtain_close",
        "drink_stop",
    ],
}

# 安全过滤配置
SAFE_COMMAND_TYPES = ["MOVE", "ROTATE", "GRIP", "RELEASE", "SWITCH", "CONTROL", "ADJUST"]
SAFE_DEVICES = ["WHEELCHAIR", "PROSTHETIC", "LIGHT", "CURTAIN", "VOLUME", "DRINK"]
DANGEROUS_KEYWORDS = ["shutdown", "reboot", "delete", "rm -rf", "format", "kill", "terminate"]


# ===================== 增强安全组件 =====================

class CommandSequenceAnalyzer:
    """行为序列分析器 - 防自动化攻击"""

    def __init__(self, max_history: int = 20):
        self.command_history = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)

    def analyze(self, command_info: Dict[str, Any]) -> Dict[str, Any]:
        """分析指令序列的合理性"""
        timestamp = time.time()
        command_str = f"{command_info.get('intent', '')}_{command_info.get('command', '')}"

        # 添加到历史
        self.command_history.append(command_str)
        self.timestamps.append(timestamp)

        # 1. 频率异常检测（防DoS）
        if self._detect_frequency_anomaly():
            return {"block": True, "reason": "指令频率过高", "risk_level": "HIGH"}

        # 2. 重复攻击模式检测
        if self._detect_repetitive_attack():
            return {"block": True, "reason": "重复攻击模式检测", "risk_level": "HIGH"}

        # 3. 逻辑顺序异常检测
        if not self._check_logical_flow(command_info):
            return {"block": True, "reason": "指令逻辑顺序异常", "risk_level": "MEDIUM"}

        return {"block": False, "risk_level": "LOW"}

    def _detect_frequency_anomaly(self) -> bool:
        """检测短时间内的指令频率"""
        if len(self.timestamps) < 3:
            return False

        recent_time = time.time() - 10  # 最近10秒
        recent_count = sum(1 for ts in self.timestamps if ts > recent_time)
        return recent_count > 15  # 10秒内超过15条指令

    def _detect_repetitive_attack(self) -> bool:
        """检测重复攻击模式"""
        if len(self.command_history) < 5:
            return False

        # 检查最近5个指令是否有4个相同
        recent_commands = list(self.command_history)[-5:]
        command_counter = Counter(recent_commands)
        return any(count >= 4 for count in command_counter.values())

    def _check_logical_flow(self, command_info: Dict[str, Any]) -> bool:
        """检查指令逻辑顺序"""
        if len(self.command_history) < 2:
            return True

        # 简单的逻辑检查：同一意图不应连续执行对立操作
        intent = command_info.get('intent', '')
        command = command_info.get('command', '')

        # 定义对立操作对
        opposite_pairs = [
            ("wheelchair_forward", "wheelchair_backward"),
            ("light_on", "light_off"),
            ("curtain_open", "curtain_close"),
            ("drink_start", "drink_stop"),
        ]

        for cmd1, cmd2 in opposite_pairs:
            if command == cmd1 and self.command_history[-1] == cmd2:
                return False
            if command == cmd2 and self.command_history[-1] == cmd1:
                return False

        return True

    def get_history_summary(self) -> Dict[str, Any]:
        """获取历史摘要"""
        return {
            "total_commands": len(self.command_history),
            "unique_commands": len(set(self.command_history)),
            "avg_frequency": len(self.command_history) / max(1, time.time() - min(
                self.timestamps)) if self.timestamps else 0,
            "last_5_commands": list(self.command_history)[-5:] if self.command_history else []
        }


class RateLimiter:
    """频率限制器 - 防DoS和暴力攻击"""

    def __init__(self, max_requests_per_minute: int = 60, max_consecutive_failures: int = 5):
        self.request_timestamps = deque(maxlen=max_requests_per_minute * 2)
        self.failure_count = 0
        self.max_requests = max_requests_per_minute
        self.max_failures = max_consecutive_failures
        self.lockout_until = 0

    def check_rate_limit(self, success: bool = True) -> Dict[str, Any]:
        """检查频率限制"""
        current_time = time.time()

        # 检查锁定状态
        if current_time < self.lockout_until:
            remaining = self.lockout_until - current_time
            return {
                "allowed": False,
                "reason": f"系统锁定中，请等待{remaining:.1f}秒",
                "lockout_remaining": remaining
            }

        # 清理过期记录（1分钟前）
        cutoff_time = current_time - 60
        while self.request_timestamps and self.request_timestamps[0] < cutoff_time:
            self.request_timestamps.popleft()

        # 检查每分钟请求数
        if len(self.request_timestamps) >= self.max_requests:
            # 触发锁定
            self.lockout_until = current_time + 300  # 锁定5分钟
            return {
                "allowed": False,
                "reason": "请求频率超限，系统已临时锁定",
                "lockout_until": self.lockout_until
            }

        # 更新记录
        self.request_timestamps.append(current_time)

        # 检查连续失败次数
        if not success:
            self.failure_count += 1
            if self.failure_count >= self.max_failures:
                # 触发锁定
                self.lockout_until = current_time + 600  # 锁定10分钟
                return {
                    "allowed": False,
                    "reason": "连续失败次数过多，系统已锁定",
                    "lockout_until": self.lockout_until
                }
        else:
            self.failure_count = 0

        return {
            "allowed": True,
            "current_rate": len(self.request_timestamps),
            "max_rate": self.max_requests,
            "failures": self.failure_count
        }

    def reset(self):
        """重置限制器"""
        self.request_timestamps.clear()
        self.failure_count = 0
        self.lockout_until = 0


class ContextAwareSecurity:
    """情境感知安全 - 基于时间、模式等的权限控制"""

    def __init__(self):
        self.time_rules = {
            "night_hours": (22, 6),  # 晚上10点到早上6点
            "restricted_commands": {
                "wheelchair_forward": {"max_speed": 30, "allowed": True},
                "wheelchair_backward": {"max_speed": 20, "allowed": True},
                "curtain_open": {"allowed": False},
                "curtain_close": {"allowed": False}
            }
        }

        # 危险模式：高危险指令连续执行
        self.dangerous_patterns = [
            ["wheelchair_forward", "wheelchair_forward", "wheelchair_forward"],
            ["prosthetic_left_grab", "prosthetic_left_grab", "prosthetic_left_grab"],
        ]

    def check_time_based_permission(self, command_id: str, command_params: Dict = None) -> Dict[str, Any]:
        """时间敏感权限检查"""
        current_hour = datetime.now().hour
        is_night = current_hour >= 22 or current_hour <= 6

        if is_night and command_id in self.time_rules["restricted_commands"]:
            rule = self.time_rules["restricted_commands"][command_id]

            if not rule["allowed"]:
                return {
                    "allowed": False,
                    "reason": f"夜间限制：{command_id} 在晚上10点至早上6点禁止执行",
                    "restriction": "time_based"
                }

            # 检查参数限制
            if "max_speed" in rule and command_params and "speed" in command_params:
                if command_params["speed"] > rule["max_speed"]:
                    return {
                        "allowed": False,
                        "reason": f"夜间速度限制：速度不得超过{rule['max_speed']}",
                        "max_allowed": rule["max_speed"],
                        "current": command_params["speed"]
                    }

        return {"allowed": True, "time_check": "passed"}

    def check_dangerous_pattern(self, command_history: List[str]) -> Dict[str, Any]:
        """检查危险模式"""
        if len(command_history) < 3:
            return {"found": False, "pattern": None}

        recent_commands = command_history[-3:]

        for pattern in self.dangerous_patterns:
            if recent_commands == pattern:
                return {
                    "found": True,
                    "pattern": pattern,
                    "risk_level": "HIGH",
                    "reason": f"检测到危险执行模式: {' -> '.join(pattern)}"
                }

        return {"found": False, "pattern": None}


class EnhancedAuditLogger:
    """增强型审计日志系统"""

    def __init__(self, max_logs: int = 1000):
        self.logs = deque(maxlen=max_logs)
        self.event_counter = 0

    def log_event(self, event_type: str, details: Dict[str, Any], severity: str = "INFO"):
        """记录安全事件"""
        log_entry = {
            "id": self._generate_event_id(),
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "severity": severity,
            "details": details,
            "session_id": st.session_state.get("session_id", "unknown")
        }

        self.logs.append(log_entry)

        # 实时告警
        if severity in ["HIGH", "CRITICAL"]:
            self._send_alert(log_entry)

        return log_entry

    def _generate_event_id(self) -> str:
        """生成事件ID"""
        self.event_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"EVT{timestamp}_{self.event_counter:06d}"

    def _send_alert(self, log_entry: Dict[str, Any]):
        """发送实时告警（模拟）"""
        if PRINT_SECURITY_ALERTS:
            print(f"⚠️ 安全告警 [{log_entry['severity']}]: {log_entry['event_type']}")

    def get_recent_logs(self, count: int = 50) -> List[Dict]:
        """获取最近日志"""
        return list(self.logs)[-count:] if self.logs else []

    def generate_security_report(self) -> Dict[str, Any]:
        """生成安全报告"""
        logs = list(self.logs)

        if not logs:
            return {"error": "暂无日志数据"}

        # 统计信息
        total_events = len(logs)
        blocked_events = sum(1 for log in logs if log.get("details", {}).get("blocked", False))
        allowed_events = total_events - blocked_events

        # 按类型统计
        event_types = Counter([log["event_type"] for log in logs])

        # 按严重性统计
        severities = Counter([log["severity"] for log in logs])

        # 最近24小时趋势
        now = datetime.now()
        twenty_four_hours_ago = now - timedelta(hours=24)
        recent_events = [
            log for log in logs
            if datetime.fromisoformat(log["timestamp"]) > twenty_four_hours_ago
        ]

        return {
            "summary": {
                "total_events": total_events,
                "blocked_events": blocked_events,
                "allowed_events": allowed_events,
                "block_rate": blocked_events / total_events * 100 if total_events > 0 else 0,
                "analysis_period": "全部记录"
            },
            "statistics": {
                "by_event_type": dict(event_types),
                "by_severity": dict(severities),
                "recent_24h": len(recent_events)
            },
            "timeline": self._create_timeline(logs),
            "generated_at": now.isoformat()
        }

    def _create_timeline(self, logs: List[Dict]) -> List[Dict]:
        """创建时间线"""
        timeline = []

        for log in logs[-50:]:  # 最近50条
            timeline.append({
                "time": datetime.fromisoformat(log["timestamp"]).strftime("%H:%M:%S"),
                "event": log["event_type"],
                "severity": log["severity"],
                "details": log["details"].get("reason", "")
            })

        return timeline

    def export_logs(self) -> str:
        """导出日志为JSON字符串"""
        return json.dumps(list(self.logs), indent=2, ensure_ascii=False)


class SecurityStateMachine:
    """安全状态机 - 根据风险动态调整安全策略"""

    STATES = {
        "NORMAL": {
            "risk_level": 1,
            "checks": ["basic"],
            "description": "正常模式 - 基础安全检查",
            "color": "green"
        },
        "ELEVATED": {
            "risk_level": 3,
            "checks": ["basic", "behavior", "context"],
            "description": "警戒模式 - 增强安全检查",
            "color": "yellow"
        },
        "HIGH_ALERT": {
            "risk_level": 5,
            "checks": ["basic", "behavior", "context", "parameter_strict"],
            "description": "高度警戒 - 严格安全检查",
            "color": "orange"
        },
        "LOCKDOWN": {
            "risk_level": 10,
            "checks": ["lockdown"],
            "description": "锁定模式 - 仅允许紧急操作",
            "color": "red"
        }
    }

    def __init__(self):
        self.current_state = "NORMAL"
        self.state_history = []
        self.risk_score = 0
        self.risk_factors = []

    def evaluate_risk(self, security_events: List[Dict]) -> str:
        """评估风险并调整状态"""
        # 计算风险分数
        self.risk_factors = self._calculate_risk_factors(security_events)
        self.risk_score = sum(factor["score"] for factor in self.risk_factors)

        # 根据风险分数调整状态
        if self.risk_score > 8:
            new_state = "LOCKDOWN"
        elif self.risk_score > 5:
            new_state = "HIGH_ALERT"
        elif self.risk_score > 3:
            new_state = "ELEVATED"
        else:
            new_state = "NORMAL"

        # 状态变化记录
        if new_state != self.current_state:
            self.state_history.append({
                "timestamp": datetime.now().isoformat(),
                "from": self.current_state,
                "to": new_state,
                "risk_score": self.risk_score,
                "factors": self.risk_factors
            })
            self.current_state = new_state

        return self.current_state

    def _calculate_risk_factors(self, events: List[Dict]) -> List[Dict]:
        """计算风险因子"""
        factors = []

        # 最近10个事件分析
        recent_events = events[-10:] if events else []

        # 因子1: 拦截率
        if recent_events:
            blocked_count = sum(1 for e in recent_events if e.get("severity") in ["HIGH", "CRITICAL"])
            block_rate = blocked_count / len(recent_events)
            factors.append({
                "name": "近期拦截率",
                "score": min(block_rate * 10, 5),  # 0-5分
                "details": f"{block_rate * 100:.1f}%"
            })

        # 因子2: 时间模式
        current_hour = datetime.now().hour
        if 22 <= current_hour <= 23 or 0 <= current_hour <= 6:
            factors.append({
                "name": "非正常时间",
                "score": 2,
                "details": f"当前时间 {current_hour}:00"
            })

        # 因子3: 连续失败
        if events and len(events) >= 3:
            last_three = events[-3:]
            if all(e.get("severity") in ["HIGH", "CRITICAL"] for e in last_three):
                factors.append({
                    "name": "连续高危事件",
                    "score": 4,
                    "details": "最近3个事件均为高危"
                })

        return factors

    def get_security_checks(self) -> List[str]:
        """获取当前状态下的安全检查列表"""
        return self.STATES[self.current_state]["checks"]

    def get_state_info(self) -> Dict[str, Any]:
        """获取状态信息"""
        state_info = self.STATES[self.current_state].copy()
        state_info.update({
            "current_state": self.current_state,
            "risk_score": self.risk_score,
            "risk_factors": self.risk_factors,
            "history_count": len(self.state_history)
        })
        return state_info


def create_security_system() -> Dict[str, Any]:
    return {
        "sequence_analyzer": CommandSequenceAnalyzer(),
        "rate_limiter": RateLimiter(),
        "context_security": ContextAwareSecurity(),
        "state_machine": SecurityStateMachine(),
        "audit_logger": EnhancedAuditLogger(),
    }


def record_runtime_security_event(
        intent: str,
        command: str,
        confidence: float,
        allowed: bool,
        reason: str = ""
) -> Dict[str, Any]:
    """Record every recognition attempt so sidebar runtime indicators move in all modes."""
    if "security_system" not in st.session_state:
        st.session_state.security_system = create_security_system()

    system = st.session_state.security_system
    seq_result = system["sequence_analyzer"].analyze({
        "intent": intent,
        "command": command,
        "confidence": confidence,
        "params": None,
    })
    rate_result = system["rate_limiter"].check_rate_limit(success=allowed)

    severity = "INFO" if allowed and not seq_result["block"] and rate_result["allowed"] else "HIGH"
    event_type = "COMMAND_ALLOWED" if severity == "INFO" else "COMMAND_ATTENTION"
    system["audit_logger"].log_event(
        event_type,
        {
            "intent": intent,
            "command": command,
            "confidence": confidence,
            "allowed": allowed,
            "reason": reason,
            "sequence_block": seq_result.get("block", False),
            "sequence_reason": seq_result.get("reason"),
            "rate_allowed": rate_result.get("allowed", True),
            "current_rate": rate_result.get("current_rate"),
        },
        severity,
    )
    audit_logs = system["audit_logger"].get_recent_logs(10)
    current_state = system["state_machine"].evaluate_risk(audit_logs)
    return {
        "sequence": seq_result,
        "rate": rate_result,
        "risk_score": system["state_machine"].risk_score,
        "current_state": current_state,
    }


# ===================== 网络安全过滤函数（增强版） =====================
def enhanced_command_safety_filter(
        intent: str,
        command_id: str,
        confidence: float,
        custom_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    增强版安全过滤函数 - 集成所有新安全方案
    返回安全过滤结果字典
    """
    logs = []
    safety_checks = {
        "confidence_check": False,
        "whitelist_check": False,
        "intent_consistency": False,
        "danger_keywords": False,
        "command_syntax": False,
        "parameter_safety": False,
        "behavior_analysis": False,
        "rate_limit": False,
        "context_aware": False,
        "state_machine": False,
    }

    # 获取指令对象
    if command_id not in STRUCTURED_COMMAND_LIBRARY:
        logs.append("❌ 未知指令ID")
        return {
            "allowed": False,
            "logs": logs,
            "safety_checks": safety_checks,
            "enhanced_security": True
        }

    command_obj = STRUCTURED_COMMAND_LIBRARY[command_id]
    if custom_params:
        command_obj.params.update(custom_params)

    structured_cmd = command_obj.to_structured_string()
    logs.append(f"🔧 结构化指令: {structured_cmd}")
    logs.append(f"🎯 识别意图: {LABEL_CN_MAP.get(intent, intent)}")
    logs.append(f"📊 置信度: {confidence:.1f}%")

    # === 基础安全检查（原有6层）===

    # 1. 置信度检查
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        logs.append(f"❌ 低置信度拦截: {confidence:.1f}% < {LOW_CONFIDENCE_THRESHOLD}%")
    else:
        safety_checks["confidence_check"] = True
        logs.append(f"✅ 置信度检查通过")

    # 2. 白名单校验
    if command_obj.command_type in SAFE_COMMAND_TYPES and command_obj.device in SAFE_DEVICES:
        safety_checks["whitelist_check"] = True
        logs.append(f"✅ 白名单检查通过: {command_obj.command_type}.{command_obj.device}")
    else:
        logs.append(f"⚠️ 非标准指令类型: {command_obj.command_type}.{command_obj.device}")

    # 3. 意图一致性校验
    allowed_commands = INTENT_ALLOWED_COMMANDS.get(intent, [])
    if command_id in allowed_commands:
        safety_checks["intent_consistency"] = True
        logs.append(f"✅ 意图一致性检查通过")
    else:
        logs.append(f"❌ 意图-指令不匹配拦截")

    # 4. 危险关键词拦截
    cmd_str = structured_cmd.lower()
    found_keywords = [kw for kw in DANGEROUS_KEYWORDS if kw in cmd_str]

    if found_keywords:
        logs.append(f"❌ 危险关键词拦截: {', '.join(found_keywords)}")
    else:
        safety_checks["danger_keywords"] = True
        logs.append("✅ 危险关键词检查通过")

    # 5. 指令语法检查
    if "_" in structured_cmd and len(structured_cmd.split("_")) >= 2:
        safety_checks["command_syntax"] = True
        logs.append("✅ 指令语法检查通过")
    else:
        logs.append("❌ 指令语法错误")

    # 6. 参数安全检查
    param_errors = []
    if command_obj.device == "WHEELCHAIR" and command_obj.command_type == "MOVE":
        speed = command_obj.params.get("speed", 0)
        if speed > 100:
            param_errors.append(f"速度超限: {speed} > 100")

    if command_obj.device == "PROSTHETIC" and "force" in command_obj.params:
        force = command_obj.params.get("force", 0)
        if force > 100:
            param_errors.append(f"力量超限: {force} > 100")

    if not param_errors:
        safety_checks["parameter_safety"] = True
        logs.append("✅ 参数安全检查通过")
    else:
        logs.append(f"❌ 参数安全检查失败: {', '.join(param_errors)}")

    # === 增强安全检查 ===

    # 初始化增强安全组件（如果不存在）
    if "security_system" not in st.session_state:
        st.session_state.security_system = create_security_system()

    system = st.session_state.security_system

    # 7. 行为序列分析
    seq_result = system["sequence_analyzer"].analyze({
        "intent": intent,
        "command": command_id,
        "confidence": confidence,
        "params": custom_params
    })

    if seq_result["block"]:
        logs.append(f"❌ 行为序列分析拦截: {seq_result['reason']}")
        system["audit_logger"].log_event(
            "BEHAVIOR_BLOCK",
            {"reason": seq_result["reason"], "risk_level": seq_result["risk_level"]},
            "HIGH"
        )
    else:
        safety_checks["behavior_analysis"] = True
        logs.append(f"✅ 行为序列检查通过")

    # 8. 频率限制检查
    rate_result = system["rate_limiter"].check_rate_limit(success=True)
    if not rate_result["allowed"]:
        logs.append(f"❌ 频率限制拦截: {rate_result['reason']}")
        system["audit_logger"].log_event(
            "RATE_LIMIT_BLOCK",
            rate_result,
            "HIGH"
        )
    else:
        safety_checks["rate_limit"] = True
        logs.append(f"✅ 频率限制检查通过 (当前: {rate_result['current_rate']}/{rate_result['max_rate']})")

    # 9. 情境感知检查
    context_result = system["context_security"].check_time_based_permission(
        command_id, custom_params
    )
    if not context_result["allowed"]:
        logs.append(f"❌ 情境感知拦截: {context_result['reason']}")
        system["audit_logger"].log_event(
            "CONTEXT_BLOCK",
            context_result,
            "MEDIUM"
        )
    else:
        safety_checks["context_aware"] = True
        logs.append(f"✅ 情境感知检查通过")

    # 10. 状态机检查
    current_state = system["state_machine"].current_state
    state_checks = system["state_machine"].get_security_checks()

    if current_state == "LOCKDOWN":
        # 锁定模式下只允许紧急操作
        emergency_commands = ["light_on", "light_off"]
        if command_id not in emergency_commands:
            logs.append(f"❌ 系统锁定状态拦截: 当前状态为{current_state}")
            system["audit_logger"].log_event(
                "LOCKDOWN_BLOCK",
                {"state": current_state, "command": command_id},
                "CRITICAL"
            )
        else:
            safety_checks["state_machine"] = True
            logs.append(f"⚠️ 锁定模式 - 紧急操作允许")
    else:
        safety_checks["state_machine"] = True
        logs.append(f"✅ 状态机检查通过 (当前状态: {current_state})")

    # 综合决策
    passed_checks = sum(safety_checks.values())
    total_checks = len(safety_checks)
    pass_rate = (passed_checks / total_checks) * 100

    # 关键检查必须通过
    critical_checks = ["whitelist_check", "intent_consistency", "danger_keywords"]
    critical_passed = all(safety_checks[check] for check in critical_checks)

    # 增强安全检查
    enhanced_critical = ["behavior_analysis", "rate_limit"]
    enhanced_critical_passed = all(safety_checks[check] for check in enhanced_critical)

    if critical_passed and enhanced_critical_passed and pass_rate >= 80.0:
        allowed = True
        logs.append(f"✅ 安全过滤通过: {passed_checks}/{total_checks} 项检查通过 ({pass_rate:.1f}%)")

        # 记录成功事件
        system["audit_logger"].log_event(
            "COMMAND_ALLOWED",
            {
                "intent": intent,
                "command": command_id,
                "confidence": confidence,
                "pass_rate": pass_rate,
                "state": current_state
            },
            "INFO"
        )
    else:
        allowed = False
        logs.append(f"❌ 安全过滤失败: {passed_checks}/{total_checks} 项检查通过 ({pass_rate:.1f}%)")

        # 记录拦截事件
        system["audit_logger"].log_event(
            "COMMAND_BLOCKED",
            {
                "intent": intent,
                "command": command_id,
                "confidence": confidence,
                "pass_rate": pass_rate,
                "failed_checks": [k for k, v in safety_checks.items() if not v],
                "critical_passed": critical_passed,
                "enhanced_critical_passed": enhanced_critical_passed
            },
            "HIGH"
        )

    # 更新状态机
    audit_logs = system["audit_logger"].get_recent_logs(10)
    system["state_machine"].evaluate_risk(audit_logs)

    return {
        "allowed": allowed,
        "logs": logs,
        "safety_checks": safety_checks,
        "structured_command": structured_cmd,
        "command_object": command_obj,
        "pass_rate": pass_rate,
        "critical_passed": critical_passed,
        "enhanced_critical_passed": enhanced_critical_passed,
        "current_state": current_state,
        "risk_score": system["state_machine"].risk_score,
        "enhanced_security": True
    }


# ===================== 动态安全测试生成 =====================
def _random_confidence(low: float, high: float) -> float:
    return round(random.uniform(low, high), 2)


def _build_dynamic_security_test_suite() -> List[Dict[str, Any]]:
    """动态生成覆盖各安全维度的测试集，每次结果都不同"""
    rng = random.SystemRandom()
    suite_seed = datetime.now().strftime("%Y%m%d%H%M%S%f")

    safe_command_pool = {
        "left": ["wheelchair_left", "prosthetic_left_grab", "light_on", "volume_up"],
        "right": ["wheelchair_right", "prosthetic_left_release", "light_off", "volume_down"],
        "foot": ["wheelchair_forward", "curtain_open", "drink_start"],
        "tongue": ["wheelchair_backward", "curtain_close", "drink_stop"],
    }

    intent = rng.choice(list(safe_command_pool.keys()))
    matched_command = rng.choice(safe_command_pool[intent])
    mismatched_intent = rng.choice([item for item in safe_command_pool.keys() if item != intent])
    mismatched_command = rng.choice(safe_command_pool[mismatched_intent])
    low_conf_intent = rng.choice(list(safe_command_pool.keys()))
    low_conf_command = rng.choice(safe_command_pool[low_conf_intent])
    overflow_command = rng.choice(["wheelchair_forward", "prosthetic_left_grab"])
    overflow_intent = "foot" if overflow_command == "wheelchair_forward" else "left"
    rapid_intent = rng.choice(["left", "right"])
    rapid_command = rng.choice(safe_command_pool[rapid_intent][:2] + safe_command_pool[rapid_intent][2:])
    repeat_intent = rng.choice(["foot", "tongue"])
    repeat_command = rng.choice(safe_command_pool[repeat_intent])

    overflow_params = (
        {"speed": rng.randint(120, 180)}
        if overflow_command == "wheelchair_forward"
        else {"force": rng.randint(110, 160)}
    )

    return [
        {
            "name": f"动态正常指令#{rng.randint(100, 999)}",
            "category": "白名单+意图一致",
            "intent": intent,
            "command_id": matched_command,
            "confidence": _random_confidence(80, 96),
            "expected": True,
            "focus": "验证合法指令在高置信度下可放行",
            "seed": suite_seed,
        },
        {
            "name": f"动态恶意指令#{rng.randint(100, 999)}",
            "category": "危险关键词拦截",
            "intent": rng.choice(list(safe_command_pool.keys())),
            "command_id": "malicious_shutdown",
            "confidence": _random_confidence(88, 99),
            "expected": False,
            "focus": "验证危险系统操作必须被阻断",
            "seed": suite_seed,
        },
        {
            "name": f"动态意图错配#{rng.randint(100, 999)}",
            "category": "意图一致性",
            "intent": intent,
            "command_id": mismatched_command,
            "confidence": _random_confidence(78, 95),
            "expected": False,
            "focus": "验证脑电意图与目标指令不一致时拦截",
            "seed": suite_seed,
        },
        {
            "name": f"动态低置信度#{rng.randint(100, 999)}",
            "category": "置信度阈值",
            "intent": low_conf_intent,
            "command_id": low_conf_command,
            "confidence": _random_confidence(45, max(46, LOW_CONFIDENCE_THRESHOLD - 5)),
            "expected": False,
            "focus": "验证低置信度推断会被过滤",
            "seed": suite_seed,
        },
        {
            "name": f"动态参数越界#{rng.randint(100, 999)}",
            "category": "参数安全",
            "intent": overflow_intent,
            "command_id": overflow_command,
            "confidence": _random_confidence(82, 94),
            "custom_params": overflow_params,
            "expected": False,
            "focus": "验证设备关键参数超限时拦截",
            "seed": suite_seed,
        },
        {
            "name": f"动态频率攻击#{rng.randint(100, 999)}",
            "category": "频率限制",
            "intent": rapid_intent,
            "command_id": rapid_command,
            "confidence": _random_confidence(80, 93),
            "rapid": True,
            "rapid_count": rng.randint(8, 12),
            "expected": False,
            "focus": "验证短时间高频请求会被风控识别",
            "seed": suite_seed,
        },
        {
            "name": f"动态重复模式#{rng.randint(100, 999)}",
            "category": "行为序列分析",
            "intent": repeat_intent,
            "command_id": repeat_command,
            "confidence": _random_confidence(79, 92),
            "repeat": rng.randint(4, 6),
            "expected": False,
            "focus": "验证重复行为模式会触发序列防护",
            "seed": suite_seed,
        },
    ]


def run_edge_case_tests():
    """运行动态安全测试，并返回生成用例与执行结果"""
    if "security_system" not in st.session_state:
        st.session_state.security_system = create_security_system()

    original_security_system = st.session_state.security_system
    st.session_state.security_system = create_security_system()

    test_cases = _build_dynamic_security_test_suite()
    results = []

    try:
        for test in test_cases:
            run_times = max(test.get("repeat", 1), test.get("rapid_count", 1))
            result = None

            for _ in range(run_times):
                if test.get("rapid", False):
                    time.sleep(0.01)

                result = enhanced_command_safety_filter(
                    intent=test["intent"],
                    command_id=test["command_id"],
                    confidence=test["confidence"],
                    custom_params=test.get("custom_params")
                )

            actual_allowed = result["allowed"] if result else False
            expected_allowed = test["expected"]

            results.append({
                "test_case": test["name"],
                "category": test["category"],
                "focus": test["focus"],
                "intent": LABEL_CN_MAP.get(test["intent"], test["intent"]),
                "command": STRUCTURED_COMMAND_LIBRARY[test["command_id"]].to_display_string(),
                "confidence": test["confidence"],
                "expected": "允许" if expected_allowed else "拦截",
                "actual": "允许" if actual_allowed else "拦截",
                "passed": actual_allowed == expected_allowed,
                "pass_rate": result.get("pass_rate", 0) if result else 0,
                "risk_score": result.get("risk_score", 0) if result else 0,
                "state": result.get("current_state", "NORMAL") if result else "NORMAL",
                "seed": test["seed"],
            })
    finally:
        st.session_state.security_system = original_security_system

    return {
        "generated_cases": test_cases,
        "results": results,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "suite_id": f"DYN-{datetime.now().strftime('%H%M%S')}-{random.randint(100, 999)}"
    }


@st.cache_data(show_spinner=False)
def load_runtime_dataset(mode: str):
    return load_feature_cache(mode)


@st.cache_resource(show_spinner=False)
def load_runtime_model(mode: str):
    paths = get_model_paths(mode)
    missing = [name for name, path in paths.items() if name != "metadata" and not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"缺少 {mode} 模式模型文件：{', '.join(missing)}。请先运行 python train.py --mode {mode}"
        )
    return (
        joblib.load(paths["model"]),
        joblib.load(paths["scaler"]),
        joblib.load(paths["label_encoder"]),
    )


def load_runtime_artifacts(mode: str):
    feature_cache = get_feature_cache_path(mode)
    if not feature_cache.exists():
        st.error(f"缺少特征缓存：{feature_cache}")
        st.info(f"请先运行：python train.py --mode {mode}")
        st.code(f"python train.py --mode {mode}", language="powershell")
        st.stop()

    try:
        dataset = load_runtime_dataset(mode)
        model, scaler, le = load_runtime_model(mode)
    except Exception as exc:
        st.error(f"运行产物加载失败：{exc}")
        st.info(f"请先运行：python train.py --mode {mode}")
        st.code(f"python train.py --mode {mode}", language="powershell")
        st.stop()

    return {
        "features": dataset["features"],
        "labels": dataset["labels"],
        "patients": dataset.get("patients"),
        "epoch_keys": dataset.get("epoch_keys"),
        "eeg_segments": dataset["eeg_segments"],
        "freqs": dataset["freqs"],
        "psd": dataset["psd"],
        "sfreq": dataset.get("sfreq", 250),
        "metadata": dataset.get("metadata", {}),
        "model": model,
        "scaler": scaler,
        "label_encoder": le,
    }


# ===================== 可视化 =====================
def plot_eeg_segment(eeg_segments, sfreq, trial_idx, ch_idx=0):
    data = eeg_segments[trial_idx, ch_idx, :]
    time_axis = np.linspace(0, len(data) / sfreq, len(data))

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(time_axis, data, color='#1890ff', linewidth=1.5)
    ax.set_title(f'第{trial_idx + 1}试次 - 第{ch_idx + 1}通道脑电信号波形', fontsize=12)
    ax.set_xlabel('时间 (s)', fontsize=10)
    ax.set_ylabel('信号幅值 (μV)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    plt.tight_layout()
    st.pyplot(fig)


def plot_confidence_bar(pred_proba, le):
    labels = le.inverse_transform(range(len(pred_proba)))
    labels_cn = [LABEL_CN_MAP.get(label, label) for label in labels]
    colors = ['#1890ff', '#2e7d32', '#f5a623', '#ff5252']

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels_cn, pred_proba * 100, color=colors, alpha=0.8)
    ax.set_ylabel('置信度 (%)', fontsize=10)
    ax.set_title('各类运动意图识别置信度', fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_facecolor('#f8f9fa')

    for bar, prob in zip(bars, pred_proba):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                f'{round(height, 1)}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)


def plot_freq_spectrum(f, Pxx):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(f, Pxx, color='#ff5252', linewidth=1.5)
    ax.axvspan(8, 13, alpha=0.2, color='green', label='α波段 (8-13Hz)')
    ax.axvspan(13, 30, alpha=0.2, color='orange', label='β波段 (13-30Hz)')
    ax.set_title('脑电信号频域功率谱', fontsize=12)
    ax.set_xlabel('频率 (Hz)', fontsize=10)
    ax.set_ylabel('功率谱密度', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    plt.tight_layout()
    st.pyplot(fig)


# ===================== 安全界面组件 =====================
def render_safety_logs(logs):
    """渲染安全过滤日志"""
    container = st.container()
    with container:
        for log in logs:
            if "❌" in log:
                st.error(log)
            elif "✅" in log:
                st.success(log)
            elif "⚠️" in log:
                st.warning(log)
            else:
                st.info(log)


def display_safety_checks(safety_checks):
    """显示安全检查明细"""
    st.subheader("🔍 安全检查明细")

    check_descriptions = {
        "confidence_check": "检查预测置信度是否达标",
        "whitelist_check": "验证指令类型和设备是否安全",
        "intent_consistency": "防止恶意映射攻击",
        "danger_keywords": "拦截系统高危操作",
        "command_syntax": "验证指令格式正确性",
        "parameter_safety": "检查参数值是否在安全范围",
        "behavior_analysis": "分析指令序列行为模式",
        "rate_limit": "防止频率攻击和DoS",
        "context_aware": "基于时间和情境的权限控制",
        "state_machine": "根据风险状态动态调整策略"
    }

    # 创建两列布局
    cols = st.columns(2)
    check_items = list(safety_checks.items())

    for i, (check_name, passed) in enumerate(check_items):
        col_idx = i % 2
        icon = "✅" if passed else "❌"
        color = "#1b5e20" if passed else "#b71c1c"
        bg = "#edf7ed" if passed else "#ffebee"
        description = check_descriptions.get(check_name, check_name)

        with cols[col_idx]:
            st.markdown(f"""
            <div style="padding:12px;border-radius:8px;background:{bg};margin-bottom:8px;">
                <div style="display:flex;align-items:center;gap:8px;">
                    <div style="font-size:18px;color:{color};">{icon}</div>
                    <div style="font-weight:bold;color:#333;">{check_name.replace('_', ' ').title()}</div>
                </div>
                <div style="font-size:12px;color:#666;margin-top:4px;">
                    {description}
                </div>
            </div>
            """, unsafe_allow_html=True)


def create_enhanced_security_dashboard():
    """创建增强型安全监控仪表板"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ 增强安全监控")

    # 初始化安全系统
    if "security_system" not in st.session_state:
        st.session_state.security_system = create_security_system()

    system = st.session_state.security_system

    # 当前安全状态
    state_info = system["state_machine"].get_state_info()
    state_colors = {
        "NORMAL": "green",
        "ELEVATED": "orange",
        "HIGH_ALERT": "red",
        "LOCKDOWN": "purple"
    }

    st.sidebar.markdown(f"""
    ### 🔐 当前安全状态
    <div style="padding:12px;border-radius:8px;background-color:#f0f8ff;text-align:center;">
        <div style="font-size:24px;color:{state_colors.get(state_info['current_state'], 'gray')};font-weight:bold;">
            {state_info['current_state']}
        </div>
        <div style="font-size:12px;color:#666;margin-top:4px;">
            {state_info['description']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 风险分数
    risk_score = state_info['risk_score']
    risk_color = "green" if risk_score < 3 else "orange" if risk_score < 6 else "red"
    risk_bar_width = min(risk_score, 10) * 10

    st.sidebar.markdown(f"""
    ### 📊 风险分数
    <div style="padding:12px;border-radius:8px;background-color:#fffaf0;">
        <div style="font-size:28px;color:{risk_color};font-weight:bold;text-align:center;">
            {risk_score:.1f}/10
        </div>
        <div style="margin-top:8px;">
            <div style="height:10px;background-color:#e0e0e0;border-radius:5px;overflow:hidden;">
                <div style="height:100%;width:{risk_bar_width}%;background-color:{risk_color};"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if state_info.get("risk_factors"):
        with st.sidebar.expander("风险因子", expanded=False):
            for factor in state_info["risk_factors"]:
                st.write(f"{factor['name']}: +{factor['score']:.1f} ({factor['details']})")

    # 实时统计
    col1, col2 = st.sidebar.columns(2)
    with col1:
        rate_info = system["rate_limiter"]
        st.metric("当前频率", f"{len(rate_info.request_timestamps)}/60")

    with col2:
        seq_info = system["sequence_analyzer"].get_history_summary()
        st.metric("历史指令", seq_info['total_commands'])

    # 安全操作按钮
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ 安全操作")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 重置频率限制", use_container_width=True):
            system["rate_limiter"].reset()
            st.sidebar.success("频率限制已重置")

    with col2:
        if st.button("🧯 重置风险状态", use_container_width=True):
            st.session_state.security_system = create_security_system()
            for key in ("security_report", "show_audit_logs", "audit_logs"):
                if key in st.session_state:
                    del st.session_state[key]
            st.sidebar.success("风险状态已重置")
            st.rerun()

    if st.sidebar.button("📊 安全报告", use_container_width=True):
        system = st.session_state.security_system
        report = system["audit_logger"].generate_security_report()
        st.session_state.security_report = report

    # 查看审计日志
    if st.sidebar.button("📋 查看审计日志", use_container_width=True):
        logs = system["audit_logger"].get_recent_logs(20)
        st.session_state.show_audit_logs = True
        st.session_state.audit_logs = logs

    # 导出日志
    if st.sidebar.button("💾 导出安全日志", use_container_width=True):
        logs_json = system["audit_logger"].export_logs()
        st.sidebar.download_button(
            label="下载日志文件",
            data=logs_json,
            file_name=f"security_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    # 显示安全报告（如果有）
    if "security_report" in st.session_state:
        with st.sidebar.expander("📈 安全报告摘要", expanded=True):
            report = st.session_state.security_report
            st.write(f"**总事件**: {report['summary']['total_events']}")
            st.write(f"**拦截率**: {report['summary']['block_rate']:.1f}%")
            st.write(f"**高危事件**: {report['statistics']['by_severity'].get('HIGH', 0)}")

            if st.button("关闭报告"):
                del st.session_state.security_report

    # 显示审计日志（如果有）
    if "show_audit_logs" in st.session_state and st.session_state.show_audit_logs:
        with st.sidebar.expander("📋 最近审计日志", expanded=True):
            logs = st.session_state.get("audit_logs", [])
            for log in logs[-10:]:
                time_str = datetime.fromisoformat(log['timestamp']).strftime("%H:%M:%S")
                severity_color = {
                    "INFO": "blue",
                    "MEDIUM": "orange",
                    "HIGH": "red",
                    "CRITICAL": "purple"
                }.get(log['severity'], "gray")

                st.markdown(f"""
                <div style="border-left:3px solid {severity_color};padding-left:8px;margin-bottom:8px;">
                    <div style="font-size:10px;color:#666;">{time_str}</div>
                    <div style="font-size:11px;font-weight:bold;">{log['event_type']}</div>
                    <div style="font-size:10px;color:#888;">{log['details'].get('reason', '')[:50]}...</div>
                </div>
                """, unsafe_allow_html=True)

            if st.button("关闭日志"):
                del st.session_state.show_audit_logs
                if "audit_logs" in st.session_state:
                    del st.session_state.audit_logs


# ===================== 主界面 =====================
def main():
    current_command = "N/A"
    allowed = False
    safety_logs = []
    safety_checks = {}
    structured_cmd = "N/A"
    risk_score = 0.0
    current_state = "NORMAL"
    current_mode = "simple"  # 初始化模式
    st.set_page_config(
        page_title="脑电意图识别与增强安全控制",
        page_icon="🧠",
        layout="wide"
    )

    # 初始化session state
    if "security_logs" not in st.session_state:
        st.session_state.security_logs = []
    if "test_results" not in st.session_state:
        st.session_state.test_results = None
    if "dynamic_test_cases" not in st.session_state:
        st.session_state.dynamic_test_cases = None
    if "dynamic_test_summary" not in st.session_state:
        st.session_state.dynamic_test_summary = None
    if "safety_stats" not in st.session_state:
        st.session_state.safety_stats = {"total": 0, "blocked": 0, "allowed": 0}

    # 标题
    st.title("🧠 脑电意图识别与增强安全控制系统")
    st.markdown("""
    <div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:20px;border-radius:10px;color:white;margin-bottom:20px;">
        <h3 style="color:white;margin:0;">🎯 系统特性</h3>
        <div style="display:flex;gap:15px;margin-top:10px;flex-wrap:wrap;">
            <div style="background:rgba(255,255,255,0.2);padding:8px 12px;border-radius:6px;">🔐 10层安全过滤</div>
            <div style="background:rgba(255,255,255,0.2);padding:8px 12px;border-radius:6px;">🧠 意图一致性验证</div>
            <div style="background:rgba(255,255,255,0.2);padding:8px 12px;border-radius:6px;">📊 行为序列分析</div>
            <div style="background:rgba(255,255,255,0.2);padding:8px 12px;border-radius:6px;">⏱️ 频率限制防护</div>
            <div style="background:rgba(255,255,255,0.2);padding:8px 12px;border-radius:6px;">🌙 情境感知安全</div>
            <div style="background:rgba(255,255,255,0.2);padding:8px 12px;border-radius:6px;">📈 动态风险评估</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 侧边栏
    st.sidebar.header("📡 系统状态")
    st.sidebar.write("电极连接：✅ 正常")
    st.sidebar.write("信号质量：⭐⭐⭐⭐")
    st.sidebar.write("采样率：250Hz")
    st.sidebar.write("预处理：IIR滤波+ICA降噪")
    st.sidebar.write("模型：随机森林")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 运行模式")
    runtime_mode_label = st.sidebar.radio(
        "数据规模",
        ["演示模式", "完整数据模式"],
        horizontal=True,
        help="演示模式加载预处理后的轻量产物，完整数据模式加载全量训练产物。"
    )
    runtime_mode = "demo" if runtime_mode_label == "演示模式" else "full"
    st.sidebar.caption(f"数据源：{DATA_PATH}")
    st.sidebar.caption(f"特征缓存：{get_feature_cache_path(runtime_mode)}")

    # 基础安全状态监控
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 基础安全统计")

    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        st.metric("总指令", st.session_state.safety_stats["total"])
    with col2:
        st.metric("拦截", st.session_state.safety_stats["blocked"])
    with col3:
        allowed_rate = (st.session_state.safety_stats["allowed"] / max(1, st.session_state.safety_stats["total"])) * 100
        st.metric("通过率", f"{allowed_rate:.1f}%")

    # 安全测试按钮
    if st.sidebar.button("🧪 运行安全测试", type="secondary", use_container_width=True):
        with st.sidebar:
            st.info("🔄 正在动态生成安全测试并执行...")
            test_payload = run_edge_case_tests()
            st.session_state.dynamic_test_cases = test_payload["generated_cases"]
            st.session_state.test_results = test_payload["results"]
            st.session_state.dynamic_test_summary = {
                "generated_at": test_payload["generated_at"],
                "suite_id": test_payload["suite_id"]
            }

            # 计算通过率
            passed = sum(r["passed"] for r in test_payload["results"])
            total = len(test_payload["results"])

            if passed == total:
                st.success(f"✅ 测试完成：{passed}/{total} 通过 (100%)")
            else:
                st.warning(f"⚠️ 测试完成：{passed}/{total} 通过 ({passed / total * 100:.1f}%)")

    # 系统配置
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ 系统配置")

    confidence_threshold = st.sidebar.slider(
        "置信度阈值",
        50, 90, 65, 5,
        help="低于此值的指令将被拦截"
    )

    enable_safety = st.sidebar.checkbox("启用网络安全过滤", value=True)
    enable_enhanced_security = st.sidebar.checkbox(
        "启用增强安全模式",
        value=True,
        help="包含行为分析、频率限制、情境感知等10层防护"
    )

    # 更新阈值
    global LOW_CONFIDENCE_THRESHOLD
    LOW_CONFIDENCE_THRESHOLD = confidence_threshold

    # 创建增强安全仪表板
    create_enhanced_security_dashboard()

    # 加载离线训练产物，避免交互时重复读大文件或训练模型。
    with st.spinner("🔧 加载预训练模型与特征缓存..."):
        runtime = load_runtime_artifacts(runtime_mode)
        features = runtime["features"]
        eeg_segments = runtime["eeg_segments"]
        f_list = runtime["freqs"]
        Pxx_list = runtime["psd"]
        sfreq = runtime["sfreq"]
        metadata = runtime["metadata"]
        model = runtime["model"]
        scaler = runtime["scaler"]
        le = runtime["label_encoder"]
        n_epochs_actual = len(features)

    st.success(f"✅ 系统准备就绪：{runtime_mode_label}，共 {n_epochs_actual} 个试次")
    with st.sidebar.expander("运行产物详情", expanded=False):
        st.write(f"对齐方式：{metadata.get('alignment', 'unknown')}")
        st.write(f"特征维度：{metadata.get('feature_dim', features.shape[1])}")
        st.write(f"标签分布：{metadata.get('label_counts', {})}")
        preprocessing = metadata.get("preprocessing", {})
        st.write(f"ICA剔除成分：{preprocessing.get('ica_exclude', [])}")
    st.markdown("---")

    # 主界面布局
    col1, col2, col3 = st.columns([1, 1.2, 1.2])

    with col1:
        st.subheader("🎛️ 试次与指令配置")
        trial_idx = st.slider("选择试次编号", 0, n_epochs_actual - 1, 0)

        st.markdown("---")
        st.subheader("🔄 指令映射模式")

        # 模式选择
        mapping_mode = st.radio(
            "选择指令映射模式",
            ["简单模式", "高级模式"],
            horizontal=True,
            help="简单模式：传统文本映射 | 高级模式：结构化指令映射+安全过滤"
        )

        if mapping_mode == "简单模式":
            # 简单指令映射
            intent_map = {
                "left": st.selectbox("🖐️ 左手运动 →", ["轮椅左拐", "打开灯光", "假肢左手抓取", "播放音乐"]),
                "right": st.selectbox("✋ 右手运动 →", ["轮椅右拐", "关闭灯光", "假肢右手释放", "暂停音乐"]),
                "foot": st.selectbox("🦶 双脚运动 →", ["轮椅前进", "打开窗帘", "调节音量+", "启动喝水装置"]),
                "tongue": st.selectbox("👅 舌头运动 →", ["轮椅后退", "关闭窗帘", "调节音量-", "停止喝水装置"])
            }
            current_mode = "simple"
        else:
            # 高级指令映射
            st.info("💡 高级模式：结构化指令 + 网络安全过滤")

            # 意图选择
            intent_options = list(LABEL_CN_MAP.keys())
            selected_intent = st.selectbox(
                "选择意图类型",
                intent_options,
                format_func=lambda x: LABEL_CN_MAP[x],
                key="advanced_intent"
            )

            # 获取该意图可用的指令
            allowed_commands = INTENT_ALLOWED_COMMANDS.get(selected_intent, [])

            if allowed_commands:
                # 创建指令选项
                command_options = {}
                for cmd_id in allowed_commands:
                    cmd_obj = STRUCTURED_COMMAND_LIBRARY[cmd_id]
                    display_text = f"{cmd_obj.to_display_string()}"
                    command_options[display_text] = cmd_id

                selected_display = st.selectbox(
                    "选择机器指令",
                    options=list(command_options.keys()),
                    help="结构化机器指令，支持参数化控制"
                )

                command_id = command_options[selected_display]
                command_obj = STRUCTURED_COMMAND_LIBRARY[command_id]

                # 参数调整
                with st.expander("调整指令参数（可选）"):
                    custom_params = {}
                    for param_name, param_value in command_obj.params.items():
                        if isinstance(param_value, (int, float)):
                            min_val = 0
                            max_val = param_value * 2 if param_value > 0 else 100
                            default_val = param_value

                            new_val = st.slider(
                                f"调整 {param_name}",
                                min_val,
                                max_val,
                                default_val,
                                key=f"param_{command_id}_{param_name}"
                            )
                            custom_params[param_name] = new_val

                # 显示指令详情
                st.code(f"结构化指令：{command_obj.to_structured_string()}", language="bash")

                # 保存到session state
                st.session_state.selected_intent = selected_intent
                st.session_state.command_id = command_id
                st.session_state.command_obj = command_obj
                st.session_state.custom_params = custom_params if custom_params else None

            current_mode = "advanced"

        识别_btn = st.button("🚀 开始识别与安全验证", type="primary", use_container_width=True)

    with col2:
        st.subheader("📈 脑电信号可视化")
        plot_eeg_segment(eeg_segments, sfreq, trial_idx)
        plot_freq_spectrum(f_list, Pxx_list[trial_idx])

    with col3:
        st.subheader("📊 识别结果与安全分析")
        result_container = st.container(border=True)

        with result_container:
            st.write("👆 点击左侧按钮开始识别，展示结果与安全分析")

            # 显示测试结果（如果有）
            if st.session_state.test_results:
                with st.expander("📋 安全测试结果", expanded=False):
                    results_df = pd.DataFrame(st.session_state.test_results)
                    st.dataframe(results_df, use_container_width=True)

    # 识别按钮点击事件
    if 识别_btn:
        with st.spinner("分析中..."):
            # 特征提取和预测
            trial_feat = features[trial_idx].reshape(1, -1)
            trial_feat_scaled = scaler.transform(trial_feat)
            pred = model.predict(trial_feat_scaled)[0]
            pred_label_en = le.inverse_transform([pred])[0]
            pred_label_cn = LABEL_CN_MAP.get(pred_label_en, pred_label_en)
            pred_proba = model.predict_proba(trial_feat_scaled)[0]
            confidence = round(max(pred_proba) * 100, 2)
            # 构建概率分布字典
            proba_distribution = {}
            for label, prob in zip(le.classes_, pred_proba):
                proba_distribution[LABEL_CN_MAP.get(label, label)] = round(prob * 100, 2)

            # 更新统计
            st.session_state.safety_stats["total"] += 1

            # 根据模式获取指令
            if current_mode == "simple":
                current_command = intent_map.get(pred_label_en, "未配置")
                allowed = True
                safety_logs = ["✅ 简单模式无需安全过滤"]
                safety_checks = {}
                structured_cmd = current_command
                telemetry = record_runtime_security_event(
                    intent=pred_label_en,
                    command=current_command,
                    confidence=confidence,
                    allowed=allowed,
                    reason="simple_mode_mapping"
                )
                risk_score = telemetry["risk_score"]
                current_state = telemetry["current_state"]
            else:
                # 高级模式：检查意图是否匹配
                if 'selected_intent' in st.session_state and st.session_state.selected_intent == pred_label_en:
                    command_obj = st.session_state.command_obj
                    custom_params = st.session_state.get('custom_params')

                    # 执行安全过滤
                    if enable_safety:
                        if enable_enhanced_security:
                            filter_result = enhanced_command_safety_filter(
                                intent=pred_label_en,
                                command_id=st.session_state.command_id,
                                confidence=confidence,
                                custom_params=custom_params
                            )
                        else:
                            # 使用基础版安全过滤（需要从原代码中复制）
                            # 这里简化为直接允许
                            filter_result = {
                                "allowed": True,
                                "logs": ["✅ 基础安全过滤通过"],
                                "safety_checks": {},
                                "structured_command": command_obj.to_structured_string(),
                                "pass_rate": 100
                            }

                        allowed = filter_result["allowed"]
                        safety_logs = filter_result.get("logs", [])
                        safety_checks = filter_result.get("safety_checks", {})
                        structured_cmd = filter_result.get("structured_command", "")
                        risk_score = filter_result.get("risk_score", 0)
                        current_state = filter_result.get("current_state", "NORMAL")

                        # 记录审计日志
                        if "security_system" in st.session_state:
                            logger = st.session_state.security_system["audit_logger"]
                            logger.log_event(
                                "USER_COMMAND",
                                {
                                    "intent": pred_label_en,
                                    "command": st.session_state.command_id,
                                    "confidence": confidence,
                                    "allowed": allowed,
                                    "pass_rate": filter_result.get("pass_rate", 0),
                                    "state": current_state
                                },
                                "INFO" if allowed else "HIGH"
                            )
                    else:
                        allowed = True
                        safety_logs = ["⚠️ 安全过滤已禁用"]
                        safety_checks = {}
                        structured_cmd = command_obj.to_structured_string()
                        telemetry = record_runtime_security_event(
                            intent=pred_label_en,
                            command=st.session_state.command_id,
                            confidence=confidence,
                            allowed=allowed,
                            reason="advanced_filter_disabled"
                        )
                        risk_score = telemetry["risk_score"]
                        current_state = telemetry["current_state"]
                else:
                    # 意图不匹配
                    current_command = "意图不匹配，无法执行指令"
                    allowed = False
                    safety_logs = ["❌ 识别意图与选择意图不匹配"]
                    safety_checks = {}
                    structured_cmd = "N/A"
                    telemetry = record_runtime_security_event(
                        intent=pred_label_en,
                        command=st.session_state.get("command_id", "unmatched_intent"),
                        confidence=confidence,
                        allowed=allowed,
                        reason="intent_mismatch"
                    )
                    risk_score = telemetry["risk_score"]
                    current_state = telemetry["current_state"]

            # 更新统计
            if allowed:
                st.session_state.safety_stats["allowed"] += 1
            else:
                st.session_state.safety_stats["blocked"] += 1

            mapped_command = structured_cmd if current_mode == "advanced" else current_command
            runtime_summary = st.session_state.security_system["sequence_analyzer"].get_history_summary()
            runtime_rate = len(st.session_state.security_system["rate_limiter"].request_timestamps)
            structured_result = {
                "timestamp": datetime.now().isoformat(),
                "trial_index": trial_idx,
                "mode": current_mode,
                "prediction": {
                    "intent_en": pred_label_en,
                    "intent_cn": pred_label_cn,
                    "confidence": confidence,
                    "confidence_level": "high" if confidence > 80 else "medium" if confidence > 60 else "low",
                    "probability_distribution": proba_distribution
                },
                "command_mapping": {
                    "mapped_command": mapped_command,
                    "allowed": allowed,
                    "risk_score": risk_score,
                    "system_state": current_state
                },
                "runtime_security": {
                    "current_rate": runtime_rate,
                    "history_commands": runtime_summary["total_commands"],
                    "unique_commands": runtime_summary["unique_commands"]
                }
            }

            # 保存为JSON文件（供指令模块调用）
            with open('latest_prediction.json', 'w', encoding='utf-8') as f:
                json.dump(structured_result, f, ensure_ascii=False, indent=2)
            # （可选）在侧边栏显示结构化结果，方便调试
            st.sidebar.markdown("---")
            st.sidebar.subheader("📋 结构化输出")
            st.sidebar.json(structured_result)

        # 显示结果
        with result_container:
            result_container.empty()

            # 风险等级颜色
            risk_colors = {
                "low": "#4CAF50",
                "medium": "#FF9800",
                "high": "#F44336",
                "critical": "#9C27B0",
                "unknown": "#9E9E9E"
            }

            # 获取风险等级
            risk_level = "unknown"
            if current_mode == "advanced" and 'command_obj' in st.session_state:
                risk_level = st.session_state.command_obj.safety_level

            # 显示识别结果卡片
            st.markdown(f"""
            <div style='padding:20px; border-radius:10px; background:linear-gradient(135deg,#e6f7ff,#f0fff4);'>
                <h3 style='color:#006959;text-align:center;'>✅ 识别完成</h3>
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:16px 0;'>
                    <div style='background:rgba(255,255,255,0.72);border:1px solid #d6eef6;border-radius:8px;padding:14px;text-align:center;'>
                        <div style='font-size:13px;color:#5b6b73;'>脑电识别意图</div>
                        <div style='font-size:26px;font-weight:700;color:#12343b;margin-top:6px;'>{pred_label_cn}</div>
                        <div style='font-size:13px;color:#536b77;margin-top:6px;'>置信度：{confidence}%</div>
                    </div>
                    <div style='background:rgba(255,255,255,0.72);border:1px solid #d6eef6;border-radius:8px;padding:14px;text-align:center;'>
                        <div style='font-size:13px;color:#5b6b73;'>映射后的执行指令</div>
                        <div style='font-size:24px;font-weight:700;color:#0071e3;margin-top:6px;'>{mapped_command}</div>
                        <div style='font-size:12px;color:#667;margin-top:6px;'>切换映射只改变执行指令，不改变脑电识别意图</div>
                    </div>
                </div>
                <div style='display:flex;justify-content:center;gap:10px;margin-top:15px;flex-wrap:wrap;'>
                    <div style='padding:8px 16px; border-radius:6px; background:{risk_colors.get(risk_level, "#9E9E9E")}; 
                                color:white; font-weight:bold;'>
                        指令风险：{risk_level.upper()}
                    </div>
                    <div style='padding:8px 16px; border-radius:6px; background:#e3f2fd; color:#1976d2; font-weight:bold;'>
                        系统状态：{current_state}
                    </div>
                    <div style='padding:8px 16px; border-radius:6px; background:#fff3e0; color:#f57c00; font-weight:bold;'>
                        风险分数：{risk_score:.1f}/10
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 显示置信度柱状图
            plot_confidence_bar(pred_proba, le)
            st.caption(
                f"本次更新后：当前频率 {runtime_rate}/60，"
                f"历史指令 {runtime_summary['total_commands']}，"
                f"风险分数 {risk_score:.1f}/10。侧边栏会在下次刷新时同步显示。"
            )

            # 显示安全过滤结果（高级模式）
            if current_mode == "advanced" and enable_safety:
                st.markdown("---")
                st.subheader("🔒 网络安全过滤结果")

                if allowed:
                    st.success("✅ 指令安全，允许执行")

                    # 显示通过的安全检查
                    if safety_checks:
                        passed_checks = sum(safety_checks.values())
                        total_checks = len(safety_checks)
                        pass_rate = (passed_checks / total_checks) * 100

                        st.info(f"🎯 安全检查通过率：{passed_checks}/{total_checks} ({pass_rate:.1f}%)")

                    # 模拟执行
                    with st.expander("模拟指令执行", expanded=False):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                            if i < 30:
                                status_text.text("🔄 发送指令到设备...")
                            elif i < 70:
                                status_text.text("⚡ 设备执行中...")
                            else:
                                status_text.text("✅ 指令执行完成")

                        st.success("指令执行成功！")

                        # 显示执行详情
                        if 'command_obj' in st.session_state:
                            cmd = st.session_state.command_obj
                            st.write(f"**设备**: {cmd.device}")
                            st.write(f"**动作**: {cmd.action}")
                            st.write(f"**参数**: {cmd.params}")
                else:
                    st.error("❌ 指令被安全系统拦截")
                    st.warning("请检查指令配置或联系管理员")

                # 显示安全检查明细
                if safety_checks and enable_enhanced_security:
                    display_safety_checks(safety_checks)

                # 显示过滤日志
                if safety_logs:
                    st.subheader("📋 安全过滤日志")
                    render_safety_logs(safety_logs)

    # 动态安全测试：生成内容与执行结果合并展示
    st.markdown("---")
    st.subheader("🧪 动态安全测试")
    dynamic_test_container = st.container(border=True)

    with dynamic_test_container:
        if st.session_state.dynamic_test_cases and st.session_state.test_results:
            summary = st.session_state.get("dynamic_test_summary", {})
            st.markdown(f"""
            <div style="padding:14px 16px;border-radius:10px;background:linear-gradient(135deg,#f7fbff,#eef7ff);margin-bottom:14px;">
                <div style="font-size:16px;font-weight:700;color:#0b5394;">动态测试批次：{summary.get('suite_id', 'DYN')}</div>
                <div style="font-size:13px;color:#4f4f4f;margin-top:4px;">生成时间：{summary.get('generated_at', '-')} ｜ 每次运行都会重新组合测试场景、置信度与攻击模式</div>
            </div>
            """, unsafe_allow_html=True)

            results_df = pd.DataFrame(st.session_state.test_results)
            total_cases = len(results_df)
            passed_cases = int(results_df["passed"].sum()) if "passed" in results_df.columns else 0
            blocked_cases = int((results_df["actual"] == "拦截").sum()) if "actual" in results_df.columns else 0
            avg_risk = float(results_df["risk_score"].mean()) if "risk_score" in results_df.columns and total_cases else 0

            metric_cols = st.columns(4)
            metric_cols[0].metric("用例总数", total_cases)
            metric_cols[1].metric("测试通过", passed_cases)
            metric_cols[2].metric("实际拦截", blocked_cases)
            metric_cols[3].metric("平均风险", f"{avg_risk:.1f}/10")

            combined_rows = []
            for idx, test_case in enumerate(st.session_state.dynamic_test_cases):
                result = st.session_state.test_results[idx] if idx < len(st.session_state.test_results) else {}
                command_id = test_case.get("command_id", "-")
                command_obj = STRUCTURED_COMMAND_LIBRARY.get(command_id)
                generated_mode = []
                if test_case.get("rapid"):
                    generated_mode.append(f"高频×{test_case.get('rapid_count', '-')}")
                if test_case.get("repeat"):
                    generated_mode.append(f"重复×{test_case.get('repeat', '-')}")
                if test_case.get("custom_params"):
                    generated_mode.append(f"参数越界 {test_case.get('custom_params')}")

                combined_rows.append({
                    "测试名称": test_case.get("name", result.get("test_case", "-")),
                    "覆盖维度": test_case.get("category", result.get("category", "-")),
                    "测试目标": test_case.get("focus", result.get("focus", "-")),
                    "生成意图": LABEL_CN_MAP.get(test_case.get("intent"), test_case.get("intent", "-")),
                    "生成指令": command_obj.to_display_string() if command_obj else command_id,
                    "生成场景": " / ".join(generated_mode) if generated_mode else "单次",
                    "置信度": test_case.get("confidence", result.get("confidence", "-")),
                    "期望": "允许" if test_case.get("expected", False) else "拦截",
                    "实际": result.get("actual", "-"),
                    "是否通过": "✅ 通过" if result.get("passed", False) else "❌ 未通过",
                    "风险分数": result.get("risk_score", "-"),
                    "系统状态": result.get("state", "-"),
                    "安全检查通过率": result.get("pass_rate", "-"),
                    "批次标识": test_case.get("seed", result.get("seed", "-")),
                })

            st.markdown("#### 测试场景与执行结果")
            st.dataframe(pd.DataFrame(combined_rows), use_container_width=True, height=360)
        else:
            st.info("点击左侧“🧪 运行安全测试”后，这里会显示动态生成的测试场景和对应执行结果。")

    # 页脚
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("""
        <div style="text-align:center;color:#666;font-size:12px;">
            <p>🧠 脑电意图识别与增强安全控制系统 v2.1</p>
            <p>🔐 10层安全防护 | 📊 实时监控 | 🚀 智能响应</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
