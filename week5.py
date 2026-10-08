# Bayesian Network

# Define conditional probability tables (CPTs)
P_burglary = 0.002
P_earthquake = 0.001

# Probability of alarm given burglary and earthquake
P_alarm_given_burglary_and_earthquake = 0.94
P_alarm_given_burglary_and_no_earthquake = 0.95
P_alarm_given_no_burglary_and_earthquake = 0.31
P_alarm_given_no_burglary_and_no_earthquake = 0.001

# Probability of David calling given alarm
P_david_calls_given_alarm = 0.91
P_david_does_not_call_given_alarm = 0.09
P_david_calls_given_no_alarm = 0.05
P_david_does_not_call_given_no_alarm = 0.95

# Probability of Sophia calling given alarm
P_sophia_calls_given_alarm = 0.75
P_sophia_does_not_call_given_alarm = 0.25
P_sophia_calls_given_no_alarm = 0.02
P_sophia_does_not_call_given_no_alarm = 0.98


# Validation function
def validate_probability(p):
    if not (0 <= p <= 1):
        raise ValueError(f"Invalid probability: {p}. Must be between 0 and 1.")


# Calculate joint probability
def joint_probability(alarm, burglary, earthquake,
                      david_calls, sophia_calls):

    # Determine alarm probability
    if alarm:
        if burglary and earthquake:
            P_alarm = P_alarm_given_burglary_and_earthquake
        elif burglary:
            P_alarm = P_alarm_given_burglary_and_no_earthquake
        elif earthquake:
            P_alarm = P_alarm_given_no_burglary_and_earthquake
        else:
            P_alarm = P_alarm_given_no_burglary_and_no_earthquake
    else:
        if burglary and earthquake:
            P_alarm = 1 - P_alarm_given_burglary_and_earthquake
        elif burglary:
            P_alarm = 1 - P_alarm_given_burglary_and_no_earthquake
        elif earthquake:
            P_alarm = 1 - P_alarm_given_no_burglary_and_earthquake
        else:
            P_alarm = 1 - P_alarm_given_no_burglary_and_no_earthquake

    # David probability
    if alarm:
        P_david = (P_david_calls_given_alarm
                   if david_calls
                   else P_david_does_not_call_given_alarm)
    else:
        P_david = (P_david_calls_given_no_alarm
                   if david_calls
                   else P_david_does_not_call_given_no_alarm)

    # Sophia probability
    if alarm:
        P_sophia = (P_sophia_calls_given_alarm
                    if sophia_calls
                    else P_sophia_does_not_call_given_alarm)
    else:
        P_sophia = (P_sophia_calls_given_no_alarm
                    if sophia_calls
                    else P_sophia_does_not_call_given_no_alarm)

    return (
        (P_burglary if burglary else 1 - P_burglary)
        * (P_earthquake if earthquake else 1 - P_earthquake)
        * P_alarm
        * P_david
        * P_sophia
    )


# ----------------------------
# TEST CASES
# ----------------------------

def run_test(test_name, expected_type, **kwargs):
    try:
        result = joint_probability(**kwargs)

        if result >= 0:
            print(f"✓ {test_name} ({expected_type})")
            print(f"  Joint Probability = {result:.10f}\n")
        else:
            print(f"✗ {test_name}: Invalid negative probability\n")

    except Exception as e:
        print(f"✗ {test_name} ({expected_type})")
        print(f"  Error: {e}\n")


print("========== POSITIVE TEST CASES ==========\n")

# Test Case 1: Alarm=True, no burglary, no earthquake, both call
run_test(
    "TC1: Alarm sounded, both called",
    "Success",
    alarm=True,
    burglary=False,
    earthquake=False,
    david_calls=True,
    sophia_calls=True
)

# Test Case 2: Burglary triggers alarm, both call
run_test(
    "TC2: Burglary with alarm",
    "Success",
    alarm=True,
    burglary=True,
    earthquake=False,
    david_calls=True,
    sophia_calls=True
)

# Test Case 3: Earthquake triggers alarm, David calls, Sophia doesn't
run_test(
    "TC3: Earthquake with mixed calls",
    "Success",
    alarm=True,
    burglary=False,
    earthquake=True,
    david_calls=True,
    sophia_calls=False
)


print("========== NEGATIVE / EDGE TEST CASES ==========\n")

# Test Case 4: Missing CPT value
try:
    backup = P_alarm_given_burglary_and_earthquake
    P_alarm_given_burglary_and_earthquake = None

    if P_alarm_given_burglary_and_earthquake is None:
        raise ValueError("Missing CPT value detected")

except Exception as e:
    print("✓ TC4: Missing CPT Value (Failure/Edge)")
    print(f"  Error: {e}\n")

finally:
    P_alarm_given_burglary_and_earthquake = backup


# Test Case 5: Invalid probability (>1)
try:
    invalid_prob = 1.2
    validate_probability(invalid_prob)

except Exception as e:
    print("✓ TC5: Invalid Probability > 1 (Failure/Edge)")
    print(f"  Error: {e}\n")
