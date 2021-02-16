//! # statest
//! Rust crate for statistical tests.
//! 
//! [![Crate](http://meritbadge.herokuapp.com/statest)](https://crates.io/crates/statest)
//! [![docs.rs](https://docs.rs/statest/badge.svg)](https://docs.rs/statest)
//! 
//! Now
//! - T test
//! - Kolmogorov–Smirnov test
//! - Chi2 test
//! 
//! is available.
//!
//! ## usage
//! ### T test
//! ```rust
//! use statest::ttest::*;
//! 
//! fn main() {
//!     let v: Vec<f32> = vec![100.2, 101.5, 98.0, 100.1, 100.9, 99.6, 98.6, 102.1, 101.4, 97.9];
//!     println!("{}", v.ttest1(102.0, 0.05, Side::One(UPLO::Upper))); // true
//! }
//! ```
//! ### Kolmogorov–Smirnov test
//! ```rust
//! use rand_distr::{StudentT, Distribution};
//! use statest::ks::*;
//! use statrs::distribution::{StudentsT, Exponential, Normal};
//! 
//! fn main() {
//!     let t = StudentT::new(1.0).unwrap();
//!     let t_vec = (0..1000).map(|_| t.sample(&mut rand::thread_rng()))
//!                          .collect::<Vec<f64>>();
//!     let tdist = StudentsT::new(0.0, 1.0, 1.0).unwrap();
//!     let ndist = Normal::new(0.0, 1.0).unwrap();
//!     let edist = Exponential::new(1.0).unwrap();
//!     println!("StudentT? {}", t_vec.ks1(&tdist, 0.05)); // true
//!     println!("Normal? {}", t_vec.ks1(&ndist, 0.05)); // false
//!     println!("Exponential? {}", t_vec.ks1(&edist, 0.05)); // false
//! }
//! ```

pub mod cast;
pub mod chi2;
pub mod ks;
pub mod normal;
pub mod numerical;
pub mod ttest;