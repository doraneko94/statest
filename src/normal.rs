//! Functions related to normal distribution.

use num_traits::Float;
use statrs::distribution::Normal;

use crate::numerical::*;

/// Calculate non-significant range of normal distribution.
pub fn normal_range<T: Float>(mu: T, sigma: T, alpha: T, bonferroni: usize) -> (T, T) {
    let mu_f64 = mu.to_f64().unwrap();
    let sigma_f64 = sigma.to_f64().unwrap();
    let nd = Normal::new(mu_f64, sigma_f64).unwrap();
    let a = (alpha / T::from(bonferroni).unwrap() / T::from(2).unwrap()).to_f64().unwrap();
    let lower = newton(&nd, a, mu_f64, 1e-7);
    let upper = newton(&nd, 1.0 - a, mu_f64, 1e-7);
    (T::from(lower).unwrap(), T::from(upper).unwrap())
}