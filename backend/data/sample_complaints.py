"""
Sample complaint data for training and demo.
Covers all 6 categories and varied severity for Indian Railways context.
"""
from typing import List, Dict, Any

SAMPLE_COMPLAINTS: List[Dict[str, Any]] = [
    # Cleanliness
    {"text": "Coach was very dirty. Toilets were filthy and stinking. No cleaning at all.", "category": "cleanliness", "severity": "high"},
    {"text": "Platform was full of garbage. Rats and insects near waiting area.", "category": "cleanliness", "severity": "medium"},
    {"text": "Seats had stains and dust. AC compartment was not cleaned properly.", "category": "cleanliness", "severity": "medium"},
    {"text": "Restroom was unusable. No water and very unhygienic.", "category": "cleanliness", "severity": "high"},
    {"text": "General coach was dirty. Litter everywhere.", "category": "cleanliness", "severity": "low"},
    # Delay issues
    {"text": "Train was delayed by 6 hours. No announcement. We were stranded at station.", "category": "delay_issues", "severity": "high"},
    {"text": "Massive delay. Reached 10 hours late. No compensation info.", "category": "delay_issues", "severity": "critical"},
    {"text": "Train delayed by 2 hours. No proper information on display.", "category": "delay_issues", "severity": "medium"},
    {"text": "Repeated delays on this route. Very frustrating experience.", "category": "delay_issues", "severity": "medium"},
    {"text": "Slight delay of 30 minutes. Could have been informed earlier.", "category": "delay_issues", "severity": "low"},
    # Staff behavior
    {"text": "TTE was rude and refused to help. Asked for bribe for seat.", "category": "staff_behavior", "severity": "high"},
    {"text": "Station staff was unhelpful. Did not guide for platform change.", "category": "staff_behavior", "severity": "medium"},
    {"text": "Conductor was abusive and used foul language with passengers.", "category": "staff_behavior", "severity": "high"},
    {"text": "Catering staff was polite but service was slow.", "category": "staff_behavior", "severity": "low"},
    {"text": "No one at enquiry to help. Staff was indifferent.", "category": "staff_behavior", "severity": "medium"},
    # Food quality
    {"text": "Food in pantry was stale. Many passengers had stomach issues.", "category": "food_quality", "severity": "high"},
    {"text": "Catering quality is poor. Overpriced and tasteless.", "category": "food_quality", "severity": "medium"},
    {"text": "No vegetarian option available. Had to go hungry.", "category": "food_quality", "severity": "medium"},
    {"text": "Water was not safe. Bad smell and color.", "category": "food_quality", "severity": "critical"},
    {"text": "Food was cold and packaging was damaged.", "category": "food_quality", "severity": "low"},
    # Safety concerns
    {"text": "Overcrowded coach. No social distancing. Safety risk.", "category": "safety_concerns", "severity": "high"},
    {"text": "Footboard travel is common. No one stopping it. Very dangerous.", "category": "safety_concerns", "severity": "critical"},
    {"text": "Emergency exit was blocked. Fire hazard.", "category": "safety_concerns", "severity": "critical"},
    {"text": "Women's coach was not safe. Eve teasing reported.", "category": "safety_concerns", "severity": "high"},
    {"text": "Broken door. Could cause accident.", "category": "safety_concerns", "severity": "high"},
    # Ticketing problems
    {"text": "IRCTC site crashed. Could not book ticket. Lost PNR.", "category": "ticketing_problems", "severity": "high"},
    {"text": "Wrong deduction from account. Refund not processed for weeks.", "category": "ticketing_problems", "severity": "high"},
    {"text": "Waiting list did not clear. No alternative offered.", "category": "ticketing_problems", "severity": "medium"},
    {"text": "Chart preparation was wrong. Seat number mismatch.", "category": "ticketing_problems", "severity": "medium"},
    {"text": "Could not get concession certificate validated at counter.", "category": "ticketing_problems", "severity": "low"},
]

# More samples for better model (duplicate categories with variation)
EXTRA_SAMPLES: List[Dict[str, Any]] = [
    {"text": "Toilet was broken and leaking. Smell was unbearable.", "category": "cleanliness", "severity": "high"},
    {"text": "Train cancelled at last moment. No alternate arrangement.", "category": "delay_issues", "severity": "critical"},
    {"text": "Guard was helpful and courteous. Good experience.", "category": "staff_behavior", "severity": "low"},
    {"text": "Food was good but delivery was very late.", "category": "food_quality", "severity": "low"},
    {"text": "Suspicious unattended luggage. No security check.", "category": "safety_concerns", "severity": "critical"},
    {"text": "Duplicate charge on card. Customer care not responding.", "category": "ticketing_problems", "severity": "high"},
    {"text": "Platform was clean but coach interior was dirty.", "category": "cleanliness", "severity": "medium"},
    {"text": "Signal failure caused 4 hour delay.", "category": "delay_issues", "severity": "high"},
    {"text": "TTE was very cooperative. Thank you.", "category": "staff_behavior", "severity": "low"},
    {"text": "Unhygienic food. Found hair in meal.", "category": "food_quality", "severity": "high"},
    {"text": "No lights in coach at night. Safety issue.", "category": "safety_concerns", "severity": "high"},
    {"text": "PNR status not updating. Confusion about reservation.", "category": "ticketing_problems", "severity": "medium"},
]

def get_training_data():
    """Return all samples for training."""
    return SAMPLE_COMPLAINTS + EXTRA_SAMPLES
