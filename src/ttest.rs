//! T test.

use ndarray::*;
use num_traits::Float;
use statrs::statistics::Statistics;
use statrs::distribution::StudentsT;

use crate::cast::*;
use crate::numerical::newton;

/// One or Two sided test.
pub enum Side {
    /// One sided test.
    One(UPLO),
    /// Two sided test.
    Two,
}

/// Test side of Side::One.
pub enum UPLO {
    Upper,
    Lower,
}

/// Trait for T test.
pub trait TTest<T: Float> {
    /// T test, which returns `true` if p-value is less than `p/2`.
    fn ttest1(&self, y: T, p: T, side: Side) -> bool;
    /// T test with Bonferroni correction.
    fn ttest1_bonferroni(&self, y: T, p: T, side: Side, n_test: usize) -> bool {
        let p = p / T::from(n_test).unwrap();
        self.ttest1(y, p, side)
    }
}

impl TTest<f64> for Vec<f64> {
    fn ttest1(&self, y: f64, p: f64, side: Side) -> bool {
        ttest1(&self, y, p, side)
    }
}

impl TTest<f32> for Vec<f32> {
    fn ttest1(&self, y: f32, p: f32, side: Side) -> bool {
        self.to_vec_f64().ttest1(y as f64, p as f64, side)
    }
}

impl<S: Data<Elem = f64>> TTest<f64> for ArrayBase<S, Ix1> {
    fn ttest1(&self, y: f64, p: f64, side: Side) -> bool {
        self.to_vec().ttest1(y, p, side)
    }
}

impl<S: Data<Elem = f32>> TTest<f32> for ArrayBase<S, Ix1> {
    fn ttest1(&self, y: f32, p: f32, side: Side) -> bool {
        self.to_vec().ttest1(y, p, side)
    }
}

/// T test which returns `true` if p-value is less than `p/2`.
pub fn ttest1(x: &[f64], y: f64, p: f64, side: Side) -> bool {
    let pval = p / 2.0;
    let n = x.len();
    let m = x.mean();
    let s2 = x.variance();
    let s2_n_sqrt = (s2 / n as f64).sqrt();
    let tdist = StudentsT::new(0.0, 1.0, (n - 1) as f64).unwrap();
    match side {
        Side::Two => {
            let up = m + newton(&tdist, 1.0 - pval, 0.0, 1e-7) * s2_n_sqrt;
            if y > up {
                return true;
            }
            let lo = m - newton(&tdist, pval, 0.0, 1e-7) * s2_n_sqrt;
            if y < lo {
                true
            } else {
                false
            }
        }
        Side::One(uplo) => match uplo {
            UPLO::Upper => {
                let up = m + newton(&tdist, 1.0 - pval, 0.0, 1e-7) * s2_n_sqrt;
                if y > up {
                    true
                } else {
                    false
                }
            }
            UPLO::Lower => {
                let lo = m - newton(&tdist, pval, 0.0, 1e-7) * s2_n_sqrt;
                if y < lo {
                    true
                } else {
                    false
                }
            }
        }
    }
}