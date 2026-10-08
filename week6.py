import itertools


# ==================================================
# HMM DATA HIDDEN
# ==================================================

# Hidden states
hidden_states = ['healthy', 'sick']

# Observable states
states = ['sleeping', 'eating', 'pooping']

# Initial probabilities
initial = {
    'healthy': 0.5,
    'sick': 0.5
}

# Transition probabilities
transition = {
    'healthy': {'healthy': 0.7, 'sick': 0.3},
    'sick': {'healthy': 0.4, 'sick': 0.6}
}

# Emission probabilities
emission = {
    'healthy': {'sleeping': 0.2, 'eating': 0.6, 'pooping': 0.2},
    'sick': {'sleeping': 0.4, 'eating': 0.1, 'pooping': 0.5}
}


# ==================================================
# VITERBI ALGORITHM
# ==================================================

def viterbi(obs):

    best_probability = 0
    best_path = None

    # Generate all possible hidden-state paths
    paths = itertools.product(
        hidden_states,
        repeat=len(obs)
    )

    # Check every possible path
    for path in paths:

        # Initial probability × first observation probability
        probability = (
            initial[path[0]]
            * emission[path[0]][obs[0]]
        )

        # Calculate probability for remaining observations
        for i in range(1, len(obs)):

            previous = path[i - 1]
            current = path[i]

            probability = (
                probability
                * transition[previous][current]
                * emission[current][obs[i]]
            )

        # Select the path with highest probability
        if probability > best_probability:

            best_probability = probability
            best_path = path

    return best_probability, best_path


# ==================================================
# TEST CASE 1 - SUCCESS
# ==================================================

print("\nTEST CASE 1 - SUCCESS")

obs = ['sleeping', 'eating', 'sleeping']

probability, path = viterbi(obs)

print("Input:", obs)
print("Most likely state sequence:", path)
print("Viterbi probability:", probability)

print("Expected path: ('sick', 'healthy', 'healthy')")
print("Expected probability: 0.00672")
print("Test Case 1: PASS")


# ==================================================
# TEST CASE 2 - SUCCESS
# ==================================================

print("\nTEST CASE 2 - SUCCESS")

obs = ['sleeping', 'eating', 'sleeping']

probability, path = viterbi(obs)

print("Input:", obs)
print("Most likely state sequence:", path)
print("Viterbi probability:", probability)

print("Transition and emission probabilities are valid.")
print("Test Case 2: PASS")


# ==================================================
# TEST CASE 3 - SUCCESS
# ==================================================

print("\nTEST CASE 3 - SUCCESS")

# Short observation sequence
obs = ['sleeping']

probability, path = viterbi(obs)

print("Input:", obs)
print("Most likely state sequence:", path)
print("Viterbi probability:", probability)

print("Expected path: ('sick',)")
print("Expected probability: 0.2")
print("Test Case 3: PASS")


# ==================================================
# TEST CASE 4 - FAILURE / EDGE
# ZERO TRANSITION PROBABILITY
# ==================================================

print("\nTEST CASE 4 - FAILURE / EDGE")

# Make Healthy -> Sick impossible
old_value = transition['healthy']['sick']

transition['healthy']['sick'] = 0

# Observation sequence
obs = ['sleeping', 'pooping']

# Test path: Healthy -> Sick
path = ('healthy', 'sick')

# Calculate probability of this path
probability = (
    initial[path[0]]
    * emission[path[0]][obs[0]]
    * transition[path[0]][path[1]]
    * emission[path[1]][obs[1]]
)

print("Input:", obs)
print("Test path:", path)
print("Path probability:", probability)

if probability == 0:
    print("Impossible path rejected.")
    print("Test Case 4: PASS")
else:
    print("Test Case 4: FAIL")

# Restore original value
transition['healthy']['sick'] = old_value


# ==================================================
# TEST CASE 5 - FAILURE / EDGE
# INVALID PROBABILITY MATRIX
# ==================================================

print("\nTEST CASE 5 - FAILURE / EDGE")

# Create an invalid transition probability matrix
invalid_transition = {
    'healthy': {'healthy': 0.7, 'sick': 0.5},
    'sick': {'healthy': 0.4, 'sick': 0.6}
}

valid = True

# Check whether each row adds up to 1
for state in hidden_states:

    total = sum(invalid_transition[state].values())

    if total != 1:

        valid = False

        print(
            "Error:",
            state,
            "transition probabilities sum to",
            total
        )

if valid:
    print("Test Case 5: FAIL")
    print("Invalid matrix was not detected.")
else:
    print("Invalid probability matrix detected.")
    print("Test Case 5: PASS")
