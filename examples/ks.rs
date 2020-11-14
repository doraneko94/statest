use rand_distr::{StudentT, Distribution};
use statest::ks::*;
use statrs::distribution::{StudentsT, Exponential, Normal};

fn main() {
    let t = StudentT::new(1.0).unwrap();
    let t_vec = (0..1000).map(|_| t.sample(&mut rand::thread_rng()))
                         .collect::<Vec<f64>>();
    let tdist = StudentsT::new(0.0, 1.0, 1.0).unwrap();
    let ndist = Normal::new(0.0, 1.0).unwrap();
    let edist = Exponential::new(1.0).unwrap();
    println!("StudentT? {}", t_vec.ks1(&tdist, 0.05));
    println!("Normal? {}", t_vec.ks1(&ndist, 0.05));
    println!("Exponential? {}", t_vec.ks1(&edist, 0.05));
}