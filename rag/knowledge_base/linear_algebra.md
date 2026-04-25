# Linear Algebra Knowledge Base

## Matrix Addition & Multiplication
- Addition rule: Matrices must be same dimension. Add corresponding elements.
- Multiplication rule: To multiply $A (m \times n)$ and $B (n \times p)$, inner dimensions must match ($n$). Result is $m \times p$. $(AB)_{ij} = \sum_{k=1}^n A_{ik} B_{kj}$

## Determinant Calculation
- 2x2: $|A| = ad - bc$
- 3x3: Use cofactor expansion along any row or column. $a(ei-fh) - b(di-fg) + c(dh-eg)$

## Matrix Inverse
- 2x2 formula: $A^{-1} = \frac{1}{|A|} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$
- Condition for invertibility: $|A| \neq 0$

## Rank and Nullity Theorem
For an $m \times n$ matrix $A$, Rank(A) + Nullity(A) = $n$ (number of columns).
Rank is dimension of column space (linearly independent columns).

## Eigenvalues and Eigenvectors
- Equation: $Av = \lambda v$, where $\lambda$ is eigenvalue, $v$ is eigenvector.
- Characteristic polynomial: Det($A - \lambda I$) = 0. Solve for $\lambda$.

## Dot Product, Cross Product, Magnitude
- Dot Product: $\vec{a} \cdot \vec{b} = a_1b_1 + a_2b_2 + a_3b_3 = |\vec{a}||\vec{b}|\cos\theta$
- Cross Product (3D only): Results in a vector orthogonal to both. Use determinant of $i, j, k$ matrix.
- Magnitude: $|\vec{a}| = \sqrt{a_1^2 + a_2^2 + a_3^2}$

## System of Linear Equations
Consistency conditions:
- Consistent (has solution): Rank[A] = Rank[A|b]
- Inconsistent (no solution): Rank[A] < Rank[A|b]
- Unique solution if Rank = number of variables. Infinitely many if Rank < number of variables.
