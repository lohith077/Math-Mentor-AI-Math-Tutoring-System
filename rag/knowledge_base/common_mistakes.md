# Common Math Mistakes

## Cancellation Errors in Fractions
Incorrectly canceling terms in a sum.
Wrong: $\frac{x^2 + 2x}{x} = x^2 + 2$
Correct: $\frac{x(x+2)}{x} = x+2$ (for $x \neq 0$)

## Sign Errors in Quadratic Formula
Forgetting that $-b$ flips the sign of $b$.
Example: if $b = -3$, $-b = 3$, not $-3$.

## Forgetting $\pm$ in Square Root Solutions
When solving $x^2 = a$ for $a > 0$, the solutions must be $x = \pm \sqrt{a}$.
Wrong: $x^2 = 9 \Rightarrow x = 3$
Correct: $x^2 = 9 \Rightarrow x = 3, -3$

## Confusing Necessary vs Sufficient Conditions
- Necessary: If A happens, B MUST happen. (A implies B)
- Sufficient: If B happens, A MIGHT happen. B is enough to guarantee A.

## Domain Restrictions
Always check original equations for:
- Denominators cannot be 0.
- Values inside real square roots cannot be negative.
- Arguments of logarithms must be strictly positive ($x > 0$).

## Off-By-One in Combinatorics
Counting from $a$ to $b$ inclusive has $b - a + 1$ items, not $b - a$.

## Not Checking if Discriminant is Negative
Before calculating square roots in the quadratic formula, check $b^2 - 4ac$. If it's negative, roots are complex (unless problem asks for complex roots, state no real solutions).

## Forgetting the Constant of Integration
Indefinite integrals must include $+ C$.
$\int 2x dx = x^2 + C$. Without $C$, it represents only one specific antiderivative.
