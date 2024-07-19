from fractions import Fraction


def calculate_probability(W_A, B_A, W_B, B_B):
    # Probability of drawing a white ball from bag A
    P_white_A = Fraction(W_A, W_A + B_A)
    # Probability of drawing a black ball from bag A
    P_black_A = Fraction(B_A, W_A + B_A)

    # Probability of drawing a black ball from bag B after adding a white ball
    P_black_after_white = Fraction(B_B, W_B + 1 + B_B)
    # Probability of drawing a black ball from bag B after adding a black ball
    P_black_after_black = Fraction(B_B + 1, W_B + B_B + 1)

    # Total probability of drawing a black ball from bag B
    P_black_B = P_white_A * P_black_after_white + P_black_A * P_black_after_black

    return P_black_B


# Given values for the problem
W_A = 5
B_A = 4
W_B = 7
B_B = 6

# Calculate the result
result = calculate_probability(W_A, B_A, W_B, B_B)

# Print the result as an irreducible fraction
print(f"{result.numerator}/{result.denominator}")
