        
from typing import List, Dict

STORIES: List[Dict] = [
    {
    "who": "student",
    "preface":
        "A professor accused a student of adding a section to an essay exam after the graded exam was returned. "
        "The student responded that the teaching assistant (TA) had overlooked that portion and unfairly marked it down. "
        "Both the professor and the student provided evidence.",
    "facts": [
        "The added section appears in slightly darker ink on the scan (professor's evidence).",
        "A photo of the student's draft on their laptop shows similar wording to the disputed section (student's evidence).",
        "The TA acknowledges grading many essays quickly that day (student's evidence).",
        "The scan metadata indicates the exam was rescanned the next morning (professor's evidence).",
    ],
    "suspicious_cues": [
        "The student is currently on academic probation for missing deadlines.",
        "The student recently received a formal warning for plagiarism in another course.",
        "The student is known among classmates for turning in work at the last minute.",
        "The student is a member of a social club recently sanctioned for violating academic rules.",
        "The student has been late to multiple classes this term.",
        "The student is a new transfer still adjusting to the grading system.",
        "The student is a well-known athlete on the basketball team.",
        "The student previously disputed a different grading decision last month.",
    ],
    "positive_cues": [
        "The student is a peer tutor in the campus writing center.",
        "The student serves as a note-taker for the disability services office.",
        "The student holds a scholarship for academic integrity.",
        "The student volunteers as a resident advisor known for fairness.",
        "The student is a member of the academic honor society.",
        "The student is a lab safety ambassador for their department.",
        "The student is a teaching assistant in another course.",
        "The student is a research assistant trusted to manage data accurately.",
        "The student is a member of the academic honor society.",
    ],
    "who": "employee",
    "preface":
        "A shift supervisor accused an employee of removing items from a stockroom without authorization. "
        "The employee responded that recent inventory mismatches were caused by scanning errors and shared access. "
        "Both sides provided evidence.",
    "facts": [
        "RFID logs show the employee's badge active near the stockroom after closing (supervisor's evidence).",
        "Two pallets were mis-scanned according to the warehouse variance report (employee's evidence).",
        "CCTV footage of the aisle is partially obscured by a stacked pallet (supervisor's evidence).",
        "The stockroom keypad code is shared by the entire shift (employee's evidence).",
    ],
    "suspicious_cues": [
        "The employee recently received a written warning for timekeeping issues.",
        "The employee's station was found with unlogged packing slips last week.",
        "The employee has been late to multiple shifts this month.",
        "The employee previously asked to borrow a company tool for personal use.",
        "The employee has complained openly about not receiving a raise.",
    ],
    "positive_cues": [
        "The employee is certified to train others on inventory control.",
        "The employee is a safety ambassador recognized for careful practices.",
        "The employee has no prior write-ups across two years of service.",
        "The employee volunteers to audit end-of-shift variance reports.",
        "The employee is trusted to open and close the facility alone on weekends.",
    ],
    "who": "student",
    "preface":
        "A principal investigator accused a graduate student of altering data points in a dataset prior to a lab meeting. "
        "The student responded that instrument glitches and autosave recoveries produced the discrepancies. "
        "Both sides provided evidence.",
    "facts": [
        "A subset of values appears rounded differently in the exported file (PI's evidence).",
        "Instrument logs show two error codes during the run (student's evidence).",
        "The dataset timestamp differs by 12 minutes from the autosave checkpoint (PI's evidence).",
        "A recovered backup contains intermediate values matching the student's notes (student's evidence).",
    ],
    "suspicious_cues": [
        "The student has struggled to meet several internal analysis deadlines.",
        "The student received an earlier caution for incomplete lab notebooks.",
        "The student frequently runs samples unsupervised late at night.",
        "The student has voiced frustration about the project not moving to publication.",
        "The student missed a required QA refresher this term.",
    ],
    "positive_cues": [
        "The student maintains the lab's data management SOPs.",
        "The student was selected to lead QA checks for instrument calibration.",
        "The student won a departmental award for research transparency.",
        "The student mentors undergraduates on good documentation practices.",
        "The student is trusted to manage data backups for the group server.",
    ],
    "who": "employee",
    "preface":
        "A store manager accused a cashier of processing fraudulent returns to extract cash. "
        "The cashier responded that a point-of-sale (POS) outage forced manual entries and delayed receipt scans. "
        "Both sides provided evidence.",
    "facts": [
        "Several returns were keyed without receipts in a short window (manager's evidence).",
        "A regional alert documented a POS outage during that period (cashier's evidence).",
        "Void slips lack full customer signatures (manager's evidence).",
        "Receipt images were uploaded later during reconciliation (cashier's evidence).",
    ],
    "suspicious_cues": [
        "The cashier recently received a warning about till shortages.",
        "The cashier asked to take draws in larger-than-usual amounts.",
        "The cashier was observed socializing with known return-abuse customers.",
        "The cashier has had prior coaching for not following refund scripts.",
        "The cashier expressed frustration about not being scheduled enough hours.",
    ],
    "positive_cues": [
        "The cashier is cross-trained to mentor new hires on refunds.",
        "The cashier has no prior shortages across quarterly audits.",
        "The cashier was recognized for catching a counterfeit bill last month.",
        "The cashier assists loss-prevention with end-of-day reconciliations.",
        "The cashier has a customer-service commendation from district.",
    ],
    "who": "employee",
    "preface":
        "A facilities supervisor accused a contractor of entering a restricted area without authorization. "
        "The contractor responded that they were escorted by a supervisor and that door logs misattribute group entries. "
        "Both sides provided evidence.",
    "facts": [
        "Badge logs show the contractor's credential opening the door at 21:07 (supervisor's evidence).",
        "Escort policy permits a single scan when a supervisor holds the door (contractor's evidence).",
        "The camera angle does not fully capture the doorway (supervisor's evidence).",
        "A maintenance ticket lists the contractor on the approved overnight roster (contractor's evidence).",
    ],
    "suspicious_cues": [
        "The contractor previously requested access to the restricted area and was denied.",
        "The contractor is new to the site and unfamiliar with local protocols.",
        "The contractor has been coached on tailgating awareness before.",
        "The contractor complained about delays in getting tool cages unlocked.",
        "The contractor was observed working past scheduled hours without a supervisor present.",
    ],
    "positive_cues": [
        "The contractor is a vetted vendor with a multi-year service record.",
        "The contractor completed site-specific security training last month.",
        "The contractor was nominated for a safety award by facilities.",
        "The contractor frequently assists supervisors with after-hours calls.",
        "The contractor's company is on the preferred vendor list for critical work.",
    ],
}]
