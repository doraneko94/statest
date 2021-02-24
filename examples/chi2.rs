use ndarray::*;
use statest::chi2::*;

fn main() {
    let x = vec![317, 168, 230, 85];
    let pi = vec![0.37, 0.22, 0.32, 0.09];
    
    let chi2f = Chi2Fit::new(&x);
    println!("=== fitness ===");
    println!("{:?}", chi2f.chi2_fit(&pi));

    let cross = arr2(&[
        [85, 60],
        [ 5, 40]
    ]);
    let chi2i = Chi2Indep::new(&cross);
    println!("=== independence ===");
    println!("p = {:?}", chi2i.chi2_indep());

    println!("=== independence (test) ===");
    println!("{:?}", cross.chi2_indep(0.05));
}