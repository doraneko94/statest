//! Numerical calculation for tests.

use num_traits::Float;
use statrs::distribution::ContinuousCDF;

/// Calculate derivative.
pub fn deriv<S, T>(dist: &T, x: S, h: S) -> S
where
    S: Float,
    T: ContinuousCDF<S, S>,
{
    let fx = dist.cdf(x);
    deriv_with_fx(dist, x, fx, h)
}

/// Calculate derivative with fx value which was already calculated.
pub fn deriv_with_fx<S, T>(dist: &T, x: S, fx: S, h: S) -> S
where
    S: Float,
    T: ContinuousCDF<S, S>,
{
    let fx_h = dist.cdf(x + h);
    (fx_h - fx) / h
}

/// Newton method to solve equations.
pub fn newton<S, T>(dist: &T, y: S, mu: S, epsilon: S) -> S 
where
    S: Float,
    T: ContinuousCDF<S, S>,
{
    let fx0 = dist.cdf(mu);
    if fx0 + epsilon >= y && fx0 - epsilon <= y {
        return mu;
    }
    let mut x = if fx0 > mu {
        mu + S::epsilon()
    } else {
        mu - S::epsilon()
    };
    loop {
        let fx = dist.cdf(x);
        if fx + epsilon >= y && fx - epsilon <= y {
            break;
        }
        let df = if x >= mu {
            deriv_with_fx(dist, x, fx, epsilon)
        } else {
            deriv_with_fx(dist, x, fx, -epsilon)
        };
        x = x + (y - fx) / df;
    }
    x
}