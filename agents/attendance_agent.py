import logging
from typing import List
from core.models import StudentRecord, RiskResult

logger = logging.getLogger(__name__)

class AttendanceAgent:
    HIGH_RISK = 75.0
    MEDIUM_RISK = 85.0
    
    def analyze(self, student: StudentRecord) -> RiskResult:
        pct = student.attendance_pct
        consec = student.consecutive_absences
        
        if pct < self.HIGH_RISK or consec >= 5:
            risk = "HIGH"
        elif pct < self.MEDIUM_RISK or consec >= 3:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        
        details = f"{student.name} has {pct}% attendance with {consec} consecutive absences."
        action = "Contact school immediately." if risk == "HIGH" else "Monitor closely."
        
        return RiskResult(
            registration=student.registration,
            student_name=student.name,
            parent_name=student.parent_name,
            parent_phone=student.parent_phone,
            dimension="attendance",
            risk_level=risk,
            risk_score=max(0, 100-pct),
            details=details,
            recommended_action=action,
            flags=[f"{pct}% attendance"]
        )
    
    def find_at_risk(self, students: List[StudentRecord]) -> List[RiskResult]:
        results = [self.analyze(s) for s in students if self.analyze(s).risk_level in ["HIGH", "MEDIUM"]]
        results.sort(key=lambda x: x.risk_score, reverse=True)
        return results