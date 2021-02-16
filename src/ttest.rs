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

pub struct TTest {
    pub n: usize,
    pub mean: f64,
    pub variance: f64,
    tdist: StudentsT,
}

impl TTest {
    pub fn new(x: &[f64]) -> Self {
        let n = x.len();
        let mean = x.mean();
        let variance = x.variance();
        let tdist = StudentsT::new(0.0, 1.0, (n - 1) as f64).unwrap();

        Self { n, mean, variance, tdist }
    }

    pub fn ttest1(&self, y: f64) -> f64 {
        let t = (y - self.mean) / (self.variance / self.n as f64).sqrt();
        newton(&self.tdist, t, 0.0, 1e-7)
    }
}

/// Trait for T test.
pub trait TTestVec<T: Float> {
    /// T test, which returns `true` if p-value is less than `p` (`Side::One(_)`) or `p/2` (`Side::Two`).
    fn ttest1(&self, y: T, p: T, side: Side) -> bool;
    /// T test with Bonferroni correction.
    fn ttest1_bonferroni(&self, y: T, p: T, side: Side, n_test: usize) -> bool {
        let p = p / T::from(n_test).unwrap();
        self.ttest1(y, p, side)
    }
}

impl TTestVec<f64> for Vec<f64> {
    fn ttest1(&self, y: f64, p: f64, side: Side) -> bool {
        ttest1(&self, y, p, side)
    }
}

impl TTestVec<f32> for Vec<f32> {
    fn ttest1(&self, y: f32, p: f32, side: Side) -> bool {
        self.to_vec_f64().ttest1(y as f64, p as f64, side)
    }
}

impl<S: Data<Elem = f64>> TTestVec<f64> for ArrayBase<S, Ix1> {
    fn ttest1(&self, y: f64, p: f64, side: Side) -> bool {
        self.to_vec().ttest1(y, p, side)
    }
}

impl<S: Data<Elem = f32>> TTestVec<f32> for ArrayBase<S, Ix1> {
    fn ttest1(&self, y: f32, p: f32, side: Side) -> bool {
        self.to_vec().ttest1(y, p, side)
    }
}

/// T test which returns `true` if p-value is less than `p` (`Side::One(_)`) or `p/2` (`Side::Two`).
pub fn ttest1(x: &[f64], y: f64, p: f64, side: Side) -> bool {
    let ttest = TTest::new(x);
    let pval = match side {
        Side::One(_) => p,
        Side::Two => p / 2.0,
    };
    let pout = ttest.ttest1(y);
    match side {
        Side::One(uplo) => {
            match uplo {
                UPLO::Lower => {
                    pout < pval
                }
                UPLO::Upper => {
                    1.0 - pout < pval
                }
            }
        }
        Side::Two => pout < pval || 1.0 - pout < pval
    }
}