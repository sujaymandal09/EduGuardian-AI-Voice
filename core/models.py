from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class StudentRecord:
    registration: str
    name: str
    parent_name: str
    parent_phone: str
    attendance_pct: float
    attendance_total: int
    attendance_attended: int
    consecutive_absences: int
    performance_grade: str
    performance_remarks: str
    behavior_incidents: int
    behavior_status: str
    language: str = "en"

@dataclass
class RiskResult:
    registration: str
    student_name: str
    parent_name: str
    parent_phone: str
    dimension: str
    risk_level: str
    risk_score: float
    details: str
    recommended_action: str
    flags: List[str] = field(default_factory=list)

@dataclass
class NotificationResult:
    success: bool
    sid: Optional[str] = None
    student_id: str = ""
    channel: str = "voice"
    error_message: Optional[str] = None

@dataclass
class CallPayload:
    to_number: str
    registration: str
    student_name: str
    parent_name: str
    dimension: str
    risk_level: str
    details: str
    recommended_action: str = ""  # ← ADD THIS LINE
    language: str = "en"