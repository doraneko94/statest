use statest::normal::*;

fn main() {
    let (lo, up): (f64, f64) = normal_range(0.0, 1.0, 0.05, 1);
    println!("N(0,1): ppf(0.025)={:.3}, ppf(0.975)={:.3}", lo, up);
    let (lo, up): (f64, f64) = normal_range(1.0, 2.0, 0.05, 1);
    println!("N(1,2): ppf(0.025)={:.3}, ppf(0.975)={:.3}", lo, up);
}