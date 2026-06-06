"""
main.py - CLI Teacher Dashboard
"""
import csv
import logging
import os
import sys
from typing import List
from dotenv import load_dotenv
from core.models import StudentRecord, CallPayload
from agents.attendance_agent import AttendanceAgent
from agents.performance_agent import PerformanceAgent
from agents.behavior_agent import BehaviorAgent
from services.twilio_voice_service import TwilioVoiceService, MockVoiceService

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def load_students(csv_path="data/students.csv") -> List[StudentRecord]:
    students = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            students.append(StudentRecord(
                registration=row["registration"].strip(),
                name=row["name"].strip(),
                parent_name=row["parent_name"].strip(),
                parent_phone=row["parent_phone"].strip(),
                attendance_pct=float(row["attendance_pct"]),
                attendance_total=int(row["attendance_total"]),
                attendance_attended=int(row["attendance_attended"]),
                consecutive_absences=int(row["consecutive_absences"]),
                performance_grade=row["performance_grade"].strip(),
                performance_remarks=row["performance_remarks"].strip(),
                behavior_incidents=int(row["behavior_incidents"]),
                behavior_status=row["behavior_status"].strip(),
                language=row.get("language", "en").strip(),
            ))
    return students

def get_voice_service():
    """Check if Twilio is configured, return appropriate service."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    
    if all([account_sid, auth_token, from_number]):
        print("✅ Twilio configured - Using REAL calls")
        return TwilioVoiceService()
    else:
        print("⚠️  Twilio NOT configured - Using MOCK calls")
        print("   To use real calls, set these environment variables:")
        print("   TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER")
        return MockVoiceService()

def main():
    print("\n" + "="*50)
    print("   ATTENDANCE GUARDIAN - TEACHER DASHBOARD")
    print("="*50)
    
    # Check Twilio status
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    if not account_sid:
        print("\n💡 TIP: Create a .env file with your Twilio credentials for real calls")
        print("   TWILIO_ACCOUNT_SID=ACxxxxxxxxxx")
        print("   TWILIO_AUTH_TOKEN=your_token")
        print("   TWILIO_FROM_NUMBER=+1xxxxxxxxxx")
    
    students = load_students()
    voice = get_voice_service()
    
    while True:
        print("\n1. Attendance Risk  2. Performance  3. Behavior  4. All  0. Exit")
        choice = input("Choice: ").strip()
        
        if choice == "0": 
            print("\n👋 Goodbye!\n")
            break
        
        dims = {"1": "attendance", "2": "performance", "3": "behavior"}
        agents = {
            "attendance": AttendanceAgent(), 
            "performance": PerformanceAgent(), 
            "behavior": BehaviorAgent()
        }
        
        to_check = ["attendance", "performance", "behavior"] if choice == "4" else [dims.get(choice)]
        
        for dim in to_check:
            if not dim: 
                continue
            
            agent = agents[dim]
            at_risk = agent.find_at_risk(students)
            
            print(f"\n{dim.upper()}: {len(at_risk)} at-risk students")
            
            # Show at-risk students
            for i, student in enumerate(at_risk, 1):
                print(f"  [{i}] {student.student_name} (Reg: {student.registration}) - {student.risk_level}")
            
            if at_risk and input("\nCall parents? (yes/no): ").lower() == "yes":
                # Let teacher choose which students to call
                print("\nSelect students to call (enter numbers separated by commas, or 'all'):")
                print("Example: 1,3,5  or  all")
                selection = input("> ").strip()
                
                if selection.lower() == "all":
                    selected = at_risk
                else:
                    try:
                        indices = [int(x.strip()) - 1 for x in selection.split(",")]
                        selected = [at_risk[i] for i in indices if 0 <= i < len(at_risk)]
                    except:
                        print("Invalid selection. Skipping calls.")
                        continue
                
                if not selected:
                    print("No students selected.")
                    continue
                
                # Create payloads with recommended_action ✅
                payloads = []
                for r in selected:
                    payload = CallPayload(
                        to_number=r.parent_phone,
                        registration=r.registration,
                        student_name=r.student_name,
                        parent_name=r.parent_name,
                        dimension=dim,
                        risk_level=r.risk_level,
                        details=r.details,
                        recommended_action=r.recommended_action,  # ✅ Include this
                        language="en"
                    )
                    payloads.append(payload)
                
                print(f"\n📞 Calling {len(payloads)} parent(s)...")
                results = voice.make_batch_calls(payloads)
                print(f"✅ Calls: {results['successful']}/{results['total']} successful")
                
                if results['failed'] > 0:
                    print("❌ Some calls failed. Check the errors above.")

if __name__ == "__main__":
    main()