import logging
from typing import List
from core.models import StudentRecord, RiskResult

logger = logging.getLogger(__name__)

class PerformanceAgent:
    HIGH_RISK = ["D", "F"]
    MEDIUM_RISK = ["C", "C-"]
    
    def analyze(self, student: StudentRecord) -> RiskResult:
        grade = student.performance_grade
        remarks = student.performance_remarks
        
        if grade in self.HIGH_RISK:
            risk = "HIGH"
        elif grade in self.MEDIUM_RISK:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        
        details = f"{student.name} has grade {grade}. {remarks}"
        action = "Urgent parent-teacher meeting." if risk == "HIGH" else "Extra support recommended."
        
        return RiskResult(
            registration=student.registration,
            student_name=student.name,
            parent_name=student.parent_name,
            parent_phone=student.parent_phone,
            dimension="performance",
            risk_level=risk,
            risk_score={"F":100,"D":80,"C":60,"C-":55,"B":30,"A":10}.get(grade, 20),
            details=details,
            recommended_action=action,
            flags=[f"Grade: {grade}"]
        )
    
    def find_at_risk(self, students: List[StudentRecord]) -> List[RiskResult]:
        results = [self.analyze(s) for s in students if self.analyze(s).risk_level in ["HIGH", "MEDIUM"]]
        results.sort(key=lambda x: x.risk_score, reverse=True)
        return results