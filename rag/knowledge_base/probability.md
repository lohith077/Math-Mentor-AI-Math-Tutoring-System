# Probability Knowledge Base

## Addition and Multiplication Rules
- Addition Rule (Mutually Exclusive): $P(A \cup B) = P(A) + P(B)$
- General Addition Rule: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
- Multiplication Rule (Independent): $P(A \cap B) = P(A) \times P(B)$
- General Multiplication Rule: $P(A \cap B) = P(A) \times P(B|A)$

## Conditional Probability & Bayes' Theorem
$P(A|B) = \frac{P(A \cap B)}{P(B)}$
Bayes' Theorem: $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$
Example: Testing for a disease. $P(D|+) = \frac{P(+|D)P(D)}{P(+|D)P(D) + P(+|D')P(D')}$

## Permutations and Combinations
- Permutations (order matters): $nPr = \frac{n!}{(n-r)!}$
- Combinations (order does not matter): $nCr = \frac{n!}{r!(n-r)!}$

## Binomial Distribution
Number of successes $k$ in $n$ independent trials with probability $p$.
- PMF: $P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$
- Mean (E[X]): $np$
- Variance: $np(1-p)$

## Expected Value
$E[X] = \sum x_i P(x_i)$ for discrete random variables.

## Independent vs Mutually Exclusive Events
- Independent: The occurrence of one event does not affect the probability of the other $P(A|B) = P(A)$.
- Mutually Exclusive: Events cannot happen at the same time $P(A \cap B) = 0$.

## Common Probability Pitfalls
- Confusing $P(A|B)$ (probability of A given B) with $P(B|A)$.
- Assuming events are independent when they are not.
- Forgetting to subtract the intersection in the general addition rule.
