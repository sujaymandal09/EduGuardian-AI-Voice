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
    recommended_action: str = ""
    language: str = "en"
    # Student academic data (populated from CSV for AI context)
    attendance_pct: Optional[float] = None
    attendance_total: Optional[int] = None
    attendance_attended: Optional[int] = None
    consecutive_absences: Optional[int] = None
    performance_grade: Optional[str] = None
    performance_remarks: Optional[str] = None
    behavior_incidents: Optional[int] = None
    behavior_status: Optional[str] = None
