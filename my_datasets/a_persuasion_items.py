
from typing import List, Dict

PAIRS: List[Dict] = [
    {
        "title": "City sales tax",
        "core": "Increase the city sales tax by 1%.",
        "sad": [
            "Special-needs infants may not receive adequate attention due to staff shortages.",
            "Low-income seniors may lack heat in winter and face severe hardship.",
            "Emergency services may lack vehicles to respond in time, risking lives."
        ],
        "anger": [
            "More individuals may submit fraudulent claims to public systems.",
            "Highway delays may worsen due to mismanaged projects, wasting time.",
            "Criminals may escape on technicalities due to errors in evidence handling."
        ],
    },
    {
        "title": "School nutrition levy",
        "core": "Levy a temporary 0.25% tax to stabilize school nutrition programs.",
        "sad": [
            "Children in food-insecure homes may miss breakfast and struggle in class.",
            "Illnesses may rise if kids skip meals and lose access to fresh foods.",
            "Local shelters may face added strain supplying snacks and lunches."
        ],
        "anger": [
            "Bad actors may divert food funds via padded invoices.",
            "Vendors may exploit loopholes to deliver lower-quality meals.",
            "Supervisors may approve orders without oversight, inviting waste."
        ],
    },
    {
        "title": "Emergency shelter bond",
        "core": "Issue a bond to expand emergency shelter capacity before winter.",
        "sad": [
            "Families may face dangerous cold nights with no place to sleep.",
            "Elderly residents may suffer exposure and medical complications.",
            "Outreach teams may be forced to turn people away."
        ],
        "anger": [
            "Fraudulent rental claims may siphon limited relief dollars.",
            "Contractors may inflate invoices for basic shelter repairs.",
            "Mismanagement may leave beds empty while people sleep outside."
        ],
    },
    {
        "title": "Public health nurse hiring",
        "core": "Fund additional public health nurse positions in clinics.",
        "sad": [
            "Infant checkups may be delayed, risking missed developmental issues.",
            "Caregivers may go without guidance for chronic conditions.",
            "Homebound seniors may not receive timely wound care."
        ],
        "anger": [
            "Insurance scams may rise without proper audits.",
            "Vendors may overbill for medical supplies.",
            "Procurement errors may stall care while paperwork sits."
        ],
    },
    {
        "title": "Heat assistance surcharge",
        "core": "Add a small seasonal surcharge to subsidize home heating for the vulnerable.",
        "sad": [
            "Seniors may have to choose between medicine and heat.",
            "Children may fall ill in cold apartments.",
            "Outreach lines may be overwhelmed by urgent calls."
        ],
        "anger": [
            "Landlords may illegally withhold heat despite payments.",
            "Fly-by-night providers may take deposits and disappear.",
            "Fraudulent applications may clog the system."
        ],
    },
    {
        "title": "Flood mitigation fund",
        "core": "Create a dedicated fund for neighborhood flood defenses.",
        "sad": [
            "Families may lose belongings to repeat basement flooding.",
            "Mold exposures may increase respiratory illness.",
            "Small businesses may shutter after another storm."
        ],
        "anger": [
            "Contractors may cut corners on drainage projects.",
            "Developers may skirt rules that increase runoff.",
            "Agencies may misallocate funds to low-priority sites."
        ],
    },
    {
        "title": "Transit reliability fee",
        "core": "Add a reliability fee to stabilize transit operations.",
        "sad": [
            "Night shift workers may be stranded when routes are cut.",
            "Medical appointments may be missed without dependable buses.",
            "Students may lose tutoring time when rides don't arrive."
        ],
        "anger": [
            "Fare skippers may strain already tight budgets.",
            "Procurement delays may leave broken buses in yards.",
            "Overtime abuse may drain funds meant for service."
        ],
    },
    {
        "title": "Library access tax",
        "core": "Modestly increase property tax to extend library hours.",
        "sad": [
            "Children without home internet may lose a safe study space.",
            "Job-seekers may miss resume help and applications.",
            "Elders may lose access to tech support and classes."
        ],
        "anger": [
            "Vendors may overcharge for e-book licenses.",
            "Grant reporting may be mishandled, risking funds.",
            "Facilities staff may sign off on poor-quality work."
        ],
    },
    {
        "title": "Road safety surcharge",
        "core": "Add a road safety surcharge to fix high-injury intersections.",
        "sad": [
            "Pedestrians may continue to be hit in known danger spots.",
            "Families may lose loved ones to preventable crashes.",
            "Ambulances may face longer response times."
        ],
        "anger": [
            "Contractors may leave work incomplete but bill in full.",
            "Traffic control devices may be stolen or vandalized.",
            "Funds may be diverted to non-safety pet projects."
        ],
    },
    {
        "title": "Affordable housing trust",
        "core": "Create a trust to preserve affordable housing units.",
        "sad": [
            "Families may be displaced from long-time neighborhoods.",
            "Children may have to change schools mid-year.",
            "Elders may move far from medical care and friends."
        ],
        "anger": [
            "Speculators may leave units empty, driving up rents.",
            "Developers may exploit subsidy rules with minimal benefit.",
            "Inspection shortcuts may allow unsafe units to pass."
        ],
    },
    {
        "title": "Ambulance upgrade fee",
        "core": "Adopt a small fee to upgrade ambulance fleets.",
        "sad": [
            "Paramedics may arrive with outdated equipment.",
            "Critical patients may wait longer for transport.",
            "Cardiac outcomes may worsen without new monitors."
        ],
        "anger": [
            "Vendors may overprice replacement parts.",
            "Inventory controls may be ignored, losing supplies.",
            "Service contracts may be padded without oversight."
        ],
    },
    {
        "title": "Mental health crisis teams",
        "core": "Fund mobile crisis teams for mental health calls.",
        "sad": [
            "People in crisis may not get timely de-escalation.",
            "Families may shoulder emergencies alone.",
            "Hospital ERs may be overwhelmed with preventable visits."
        ],
        "anger": [
            "Fly-by-night providers may bill for no-show staff.",
            "Case files may be mishandled, breaching privacy.",
            "Supervision lapses may leave calls unanswered."
        ],
    },
    {
        "title": "Food inspection staffing",
        "core": "Increase food inspection staffing across districts.",
        "sad": [
            "Food-borne illness may rise from missed inspections.",
            "Small eateries may close after preventable outbreaks.",
            "Children may be hit hardest by unsafe food."
        ],
        "anger": [
            "Inspectors may accept gifts in exchange for quick passes.",
            "Reports may be left unfiled, enabling repeat offenders.",
            "Vendors may falsify temperature logs."
        ],
    },
    {
        "title": "Parks maintenance levy",
        "core": "Levy funds for basic park maintenance and lighting.",
        "sad": [
            "Children may lose safe spaces to play after school.",
            "Elders may avoid unlit paths at dusk.",
            "After-school programs may be canceled for safety."
        ],
        "anger": [
            "Contractors may bill for trash pickup they didn’t do.",
            "Graffiti removal budgets may be siphoned elsewhere.",
            "Equipment purchases may go to favored vendors."
        ],
    },
    {
        "title": "Bridge repair assessment",
        "core": "Add a temporary assessment to repair aging bridges.",
        "sad": [
            "School buses may face detours adding long delays.",
            "Ambulances may need risky alternate routes.",
            "Workers may lose hours to closures."
        ],
        "anger": [
            "Construction firms may collude to inflate bids.",
            "Inspections may be rushed to meet a deadline.",
            "Materials may be substituted without approval."
        ],
    },
    {
        "title": "Clinic weekend hours",
        "core": "Fund weekend hours at community clinics.",
        "sad": [
            "Working parents may miss necessary vaccinations.",
            "Chronic conditions may worsen without access.",
            "Infants may not get timely weight checks."
        ],
        "anger": [
            "Appointment slots may be hoarded by no-show brokers.",
            "Billing errors may misdirect refunds.",
            "Scheduling software may be neglected, causing chaos."
        ],
    },
    {
        "title": "Senior transit vouchers",
        "core": "Provide senior transit vouchers for medical trips.",
        "sad": [
            "Missed appointments may lead to severe complications.",
            "Elders may lose independence and social contact.",
            "Caregivers may be unable to assist during work hours."
        ],
        "anger": [
            "Rideshare drivers may defraud voucher systems.",
            "Dispatchers may cancel rides without accountability.",
            "Agencies may misreport ride counts for extra funding."
        ],
    },
    {
        "title": "Home weatherization fund",
        "core": "Create a fund for home weatherization assistance.",
        "sad": [
            "Families may struggle with winter utility bills.",
            "Asthma in children may worsen with drafts and mold.",
            "Low-income households may live in unhealthy homes."
        ],
        "anger": [
            "Contractors may install substandard insulation.",
            "Inspectors may sign off without visiting sites.",
            "Applications may be 'lost' to reduce workload."
        ],
    },
    {
        "title": "Digital divide bond",
        "core": "Issue a bond to expand broadband for underserved areas.",
        "sad": [
            "Students may fall behind without reliable internet.",
            "Job applicants may miss online postings.",
            "Telehealth visits may be impossible for rural elders."
        ],
        "anger": [
            "ISPs may accept subsidies without building out.",
            "Contracts may be steered to favored firms.",
            "Speed tests may be falsified to claim success."
        ],
    },
    {
        "title": "Disaster readiness training",
        "core": "Fund neighborhood disaster-readiness programs.",
        "sad": [
            "Families may lack basic plans for reunification.",
            "Medication-dependent residents may run out during storms.",
            "Pets may be abandoned in chaotic evacuations."
        ],
        "anger": [
            "Supplies may be sold off privately.",
            "Roster padding may hide no-show trainers.",
            "Equipment may be diverted to unrelated projects."
        ],
    },
]