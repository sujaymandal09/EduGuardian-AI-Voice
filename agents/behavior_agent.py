import logging
from typing import List
from core.models import StudentRecord, RiskResult

logger = logging.getLogger(__name__)

class BehaviorAgent:
    HIGH_RISK_STATUS = ["Critical"]
    MEDIUM_RISK_STATUS = ["Warning"]
    
    def analyze(self, student: StudentRecord) -> RiskResult:
        incidents = student.behavior_incidents
        status = student.behavior_status
        
        if status in self.HIGH_RISK_STATUS or incidents >= 5:
            risk = "HIGH"
        elif status in self.MEDIUM_RISK_STATUS or incidents >= 2:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        
        details = f"{student.name} has {incidents} behavioral incidents. Status: {status}"
        action = "Mandatory parent-principal meeting." if risk == "HIGH" else "Counselor meeting recommended."
        
        return RiskResult(
            registration=student.registration,
            student_name=student.name,
            parent_name=student.parent_name,
            parent_phone=student.parent_phone,
            dimension="behavior",
            risk_level=risk,
            risk_score=min(100, incidents*15),
            details=details,
            recommended_action=action,
            flags=[f"Status: {status}"]
        )
    
    def find_at_risk(self, students: List[StudentRecord]) -> List[RiskResult]:
        results = [self.analyze(s) for s in students if self.analyze(s).risk_level in ["HIGH", "MEDIUM"]]
        results.sort(key=lambda x: x.risk_score, reverse=True)
        return results