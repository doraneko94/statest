//! Kolmogorov–Smirnov test.

use ndarray::*;
use num_traits::Float;
use statrs::distribution::Univariate;

use crate::cast::*;

/// Max iteration when calculating Q_KS.
const MAXJ:usize = 100;
/// Small value 1.
const EPS1:f64 = 1.0e-3;
/// Small value 2.
const EPS2:f64 = 1.0e-8;

/// Struct for Kolmogorov–Smirnov test.
pub struct KSTest {
    pub n: usize,
    x: Vec<f64>,
}

impl KSTest {
    pub fn new(x: &[f64]) -> Self {
        let n = x.len();
        let mut x = x.to_vec();
        x.sort_by(|&a, b| a.partial_cmp(b).unwrap());
        Self { n, x }
    }

    /// Kolmogorov–Smirnov test which returns probability `prob` and test statistic D `d`. 
    pub fn ks1<T: Univariate<f64, f64>>(&self, dist: &T) -> (f64, f64) {
        let mut d = 0.0;
        let mut f_old = 0.0;

        for (i, &xi) in self.x.iter().enumerate() {
            let f_new = (i + 1) as f64 / self.n as f64;
            let f_dist = dist.cdf(xi);
            let (df_old, df_new) = ((f_old - f_dist).abs(), (f_new - f_dist).abs());
            let dt = if df_old > df_new {
                df_old
            } else {
                df_new
            };
            if dt > d {
                d = dt;
            }
            f_old = f_new;
        }
        let n_sqrt = (self.n as f64).sqrt();
        let prob = q_ks((n_sqrt + 0.12 + 0.11 / n_sqrt) * d);
        (prob, d)
    }
}

/// Trait for Kolmogorov–Smirnov test.
pub trait KSVec<T: Float> {
    /// Kolmogorov–Smirnov test. If `p` <= `prob`, then returns `true`.
    fn ks1<S: Univariate<f64, f64>>(&self, dist: &S, p: T) -> bool;
}

impl KSVec<f64> for Vec<f64> {
    fn ks1<S: Univariate<f64, f64>>(&self, dist: &S, p: f64) -> bool {
        let ks = KSTest::new(&self);
        let (prob, _) = ks.ks1(dist);
        prob >= p
    }
}

impl KSVec<f32> for Vec<f32> {
    fn ks1<S: Univariate<f64, f64>>(&self, dist: &S, p: f32) -> bool {
        self.to_vec_f64().ks1(dist, p as f64)
    }
}

impl<U: Data<Elem = f64>> KSVec<f64> for ArrayBase<U, Ix1> {
    fn ks1<S: Univariate<f64, f64>>(&self, dist: &S, p: f64) -> bool {
        self.to_vec().ks1(dist, p)
    }
}

impl<U: Data<Elem = f32>> KSVec<f32> for ArrayBase<U, Ix1> {
    fn ks1<S: Univariate<f64, f64>>(&self, dist: &S, p: f32) -> bool {
        self.to_vec().ks1(dist, p)
    }
}

/// Calculate test statistic Q_KS.
fn q_ks(lambda: f64) -> f64 {
    let mut sum = 0.0;
    let mut term_old = 0.0;
    let mut fac = 2.0;
    let m2lambda2 = -2.0 * lambda * lambda;
    for i in 1..=MAXJ {
        let j = i as f64;
        let term = fac * (m2lambda2 * j * j).exp();
        sum += term;
        if term.abs() <= EPS1 * term_old || term.abs() <= EPS2 * sum {
            return sum;
        }
        fac = -fac;
        term_old = term.abs();
    }
    1.0
}