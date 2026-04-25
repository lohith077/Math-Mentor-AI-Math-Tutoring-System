# Calculus Knowledge Base

## Limit Definition and L'Hôpital's Rule
- Formal limit definition (epsilon-delta)
- L'Hôpital's Rule: If $\lim \frac{f(x)}{g(x)}$ is of indeterminate form $0/0$ or $\infty/\infty$, then $\lim \frac{f(x)}{g(x)} = \lim \frac{f'(x)}{g'(x)}$, provided the latter limit exists.

## Standard Limit Forms
- $\lim_{x \to 0} \frac{\sin x}{x} = 1$
- $\lim_{x \to \infty} (1+\frac{1}{x})^x = e$
- $\lim_{x \to 0} \frac{e^x - 1}{x} = 1$

## Differentiation Rules
- Power Rule: $\frac{d}{dx} x^n = n x^{n-1}$
- Product Rule: $(uv)' = u'v + uv'$
- Quotient Rule: $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$
- Chain Rule: $\frac{d}{dx} [f(g(x))] = f'(g(x))g'(x)$

## Standard Derivatives Table
- $\frac{d}{dx} \sin x = \cos x$
- $\frac{d}{dx} \cos x = -\sin x$
- $\frac{d}{dx} \tan x = \sec^2 x$
- $\frac{d}{dx} e^x = e^x$
- $\frac{d}{dx} \ln x = \frac{1}{x}$

## Integration
- By Substitution: Let $u = g(x)$, $du = g'(x)dx$. $\int f(g(x))g'(x)dx = \int f(u)du$
- By Parts: $\int u dv = uv - \int v du$

## Definite Integral Properties
- $\int_a^b f(x)dx = F(b) - F(a)$ (Fundamental Theorem of Calculus)
- $\int_a^b f(x)dx = -\int_b^a f(x)dx$
- $\int_a^c f(x)dx + \int_c^b f(x)dx = \int_a^b f(x)dx$

## First and Second Derivative Tests for Optimization
- First Derivative Test: If $f'$ changes from + to - at $c$, local max. If - to +, local min.
- Second Derivative Test: If $f'(c)=0$ and $f''(c) < 0$, local max. If $f''(c) > 0$, local min.

## Critical Point Analysis
Critical points occur where $f'(x) = 0$ or $f'(x)$ is undefined. These are candidates for local extrema.
