use statest::ttest::*;

fn main() {
    let v: Vec<f32> = vec![100.2, 101.5, 98.0, 100.1, 100.9, 99.6, 98.6, 102.1, 101.4, 97.9];
    println!("{}", v.ttest1(102.0, 0.05, Side::One(UPLO::Upper)));
}