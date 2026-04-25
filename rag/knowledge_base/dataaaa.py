import os

content = """# JEE Mathematics: Comprehensive RAG Knowledge Base

---
## [Topic: Algebra]
### 1. Quadratic Equations & Identities
**Core Formulas:**
- Standard Form: $$ax^2 + bx + c = 0$$
- Roots: $$x = \\frac{-b \\pm \\sqrt{D}}{2a}$$ where $$D = b^2 - 4ac$$
- Sum of roots (S): $$\\alpha + \\beta = -b/a$$
- Product of roots (P): $$\\alpha\\beta = c/a$$

**Solution Template: Nature of Roots**
1. Calculate $D$.
2. If $D > 0$, roots are real and distinct.
3. If $D = 0$, roots are real and equal ($x = -b/2a$).
4. If $D < 0$, roots are complex conjugates.

**Common Mistakes & Pitfalls:**
- **The Square Root Trap:** Assuming $\\sqrt{x^2} = x$. Correct: $\\sqrt{x^2} = |x|$.
- **Coefficient Check:** Forgetting to check if $a \\neq 0$ before applying quadratic formulas.
- **Sign Errors:** Confusing $\\alpha + \\beta = -b/a$ with $+b/a$.

**Unit/Domain Constraints:**
- For $\\log_a(x)$ in algebraic equations: $x > 0, a > 0, a \\neq 1$.
- For $\\sqrt{f(x)}$: $f(x) \\ge 0$.

---
## [Topic: Probability]
### 1. Conditional Probability & Bayes
**Core Formulas:**
- Conditional Probability: $$P(A|B) = \\frac{P(A \\cap B)}{P(B)}$$
- Bayes' Theorem: $$P(E_i|A) = \\frac{P(A|E_i)P(E_i)}{\\sum_{j=1}^{n} P(A|E_j)P(E_j)}$$
- Binomial Distribution: $$P(X=r) = \\binom{n}{r} p^r q^{n-r}$$

**Common Mistakes & Pitfalls:**
- **Independence vs. Mutually Exclusive:** Independent means $P(A \\cap B) = P(A)P(B)$. Mutually exclusive means $P(A \\cap B) = 0$. They are NOT the same.
- **Complement Rule:** Forgetting that $P(\\text{at least one}) = 1 - P(\\text{none})$.
- **Sample Space:** Not reducing the denominator in "without replacement" problems.

**Unit/Domain Constraints:**
- $$0 \\le P(A) \\le 1$$
- Total Probability $\\sum P(E_i) = 1$.

---
## [Topic: Basic Calculus]
### 1. Limits, Derivatives & Optimization
**Core Formulas:**
- Standard Limit: $$\\lim_{x \\to 0} \\frac{\\sin x}{x} = 1$$
- L'Hôpital's Rule: If form is $0/0$ or $\\infty/\\infty$, then $$\\lim \\frac{f(x)}{g(x)} = \\lim \\frac{f'(x)}{g'(x)}$$.
- Derivative of $\\ln|x|$: $$1/x$$.

**Solution Template: Optimization (Maxima/Minima)**
1. Find $f'(x)$ and set to $0$ to find critical points.
2. Calculate $f''(x)$ at critical points.
3. **Condition:** If $f''(x) < 0$, it is a Local Maxima. If $f''(x) > 0$, it is a Local Minima.

**Common Mistakes & Pitfalls:**
- **Boundary Values:** In optimization, forgetting to check the end-points of a closed interval $[a, b]$.
- **Chain Rule:** Forgetting to differentiate the inner function (e.g., $d/dx(\\sin(x^2)) = 2x\\cos(x^2)$).

---
## [Topic: Linear Algebra Basics]
### 1. Matrices & Determinants
**Core Formulas:**
- Inverse: $$A^{-1} = \\frac{1}{|A|} \\text{adj}(A)$$
- Determinant Property: $$|AB| = |A||B|$$
- Adjoint Property: $$|\\text{adj}(A)| = |A|^{n-1}$$

**Solution Template: Cramer’s Rule**
- If $\\Delta \\neq 0$: Unique Solution.
- If $\\Delta = 0$ and any $\\Delta_i \\neq 0$: No Solution.
- If $\\Delta = 0$ and all $\\Delta_i = 0$: Infinite/No Solution (Check plane equations).

**Common Mistakes & Pitfalls:**
- **Scalar Multiplication:** Forgetting that $|kA| = k^n|A|$, where $n$ is the order of the matrix.
- **Commutativity:** Assuming $AB = BA$. This is generally false for matrices.
"""

file_name = "JEE_Math_RAG_Knowledge_Base.md"

try:
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Successfully created {file_name} in your current directory.")
except Exception as e:
    print(f"An error occurred: {e}")