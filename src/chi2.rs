use ndarray::*;
use num_traits::Float;
use statrs::distribution::{ChiSquared, ContinuousCDF};

use crate::cast::*;

pub struct Chi2Fit {
    x: Vec<usize>,
    pub n: usize,
    pub k: usize,
    dist: ChiSquared,
}

pub struct Chi2Indep {
    cross: Array2<usize>,
    xi: Array1<usize>,
    xj: Array1<usize>,
    pub n: usize,
    pub r: usize,
    pub c: usize,
    dist: ChiSquared,
}

impl Chi2Fit {
    pub fn new(x: &[usize]) -> Self {
        let x = x.to_owned();
        let n = x.iter().sum();
        let k = x.len();
        let dist = ChiSquared::new(k as f64 - 1.0).unwrap();
        Self { x, n, k, dist }
    }

    pub fn chi2_fit(&self, pi: &[f64]) -> f64 {
        if self.k != pi.len() {
            panic!()
        }
        let q = self.x.iter()
                    .zip(pi.iter())
                    .map(|(&xi, pii)| {
                        let npii = self.n as f64 * pii;
                        let xnpii = xi as f64 - npii;
                        xnpii * xnpii / npii
                    })
                    .sum::<f64>();
        1.0 - self.dist.cdf(q)
    }
}

impl Chi2Indep {
    pub fn new<S: Data<Elem = usize>>(cross: &ArrayBase<S, Ix2>) -> Self {
        let cross = cross.to_owned();
        let xi = cross.sum_axis(Axis(1));
        let xj = cross.sum_axis(Axis(0));
        let n = xi.sum_axis(Axis(0)).into_scalar();
        let r = xi.len();
        let c = xj.len();
        let dist = ChiSquared::new(((r - 1) * (c - 1)) as f64).unwrap();

        Self { cross, xi, xj, n, r, c, dist }
    }

    pub fn chi2_indep(&self) -> f64 {
        let mut q_mat = Array2::zeros((self.r, self.c));
        Zip::from(&mut q_mat)
            .and(&self.cross)
            .and(&self.xi.broadcast((self.c, self.r)).unwrap().t())
            .and(&self.xj.broadcast((self.r, self.c)).unwrap())
            .apply(|a, &b, &c, &d| {
                let ijn = (c * d) as f64 / self.n as f64;
                let nume = b as f64 - ijn;
                *a = nume * nume / ijn;
            });
        let q = q_mat.sum();
        self.dist.cdf(q)
    }
}

/// Trait for Chi2 test (Goodness of fitting).
pub trait Chi2Vec<T: Float> {
    /// Chi2 test for goodness of fit, which returns `true` if p-value is larger than `p`.
    fn chi2_fit(&self, pi: &[T], pi: T) -> bool;
}

/// Trait for Chi2 test (Independence).
pub trait Chi2Array<T: Float> {
    /// Chi2 test for independence, which returns `true` if p-value is less than `p`.
    fn chi2_indep(&self, p: T) -> bool;
}

impl Chi2Vec<f64> for Vec<usize> {
    fn chi2_fit(&self, pi: &[f64], p: f64) -> bool {
        chi2_fit(&self, pi, p)
    }
}

impl Chi2Vec<f32> for Vec<usize> {
    fn chi2_fit(&self, pi: &[f32], p: f32) -> bool {
        chi2_fit(&self, &pi.to_vec_f64(), p as f64)
    }
}

impl<S: Data<Elem = usize>> Chi2Vec<f64> for ArrayBase<S, Ix1> {
    fn chi2_fit(&self, pi: &[f64], p: f64) -> bool {
        self.to_vec().chi2_fit(pi, p)
    }
}

impl<S: Data<Elem = usize>> Chi2Vec<f32> for ArrayBase<S, Ix1> {
    fn chi2_fit(&self, pi: &[f32], p: f32) -> bool {
        self.to_vec().chi2_fit(pi, p)
    }
}

impl<S: Data<Elem = usize>, T: Float> Chi2Array<T> for ArrayBase<S, Ix2> {
    fn chi2_indep(&self, p: T) -> bool {
        chi2_indep(&self, p.to_f64().unwrap())
    }
}

/// Chi2 test for goodness of fit, which returns `true` if p-value is larger than `p`.
fn chi2_fit(x: &[usize], pi: &[f64], p: f64) -> bool {
    let chi2f = Chi2Fit::new(x);
    if 1.0 - chi2f.chi2_fit(pi) < p {
        false
    } else {
        true
    }
}

/// Chi2 test for independence, which returns `true` if p-value is less than `p`.
fn chi2_indep<S: Data<Elem = usize>>(cross: &ArrayBase<S, Ix2>, p: f64) -> bool {
    let chi2i = Chi2Indep::new(cross);
    if 1.0 - chi2i.chi2_indep() < p {
        true
    } else {
        false
    }
}