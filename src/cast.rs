//! Cast f32 -> f64.

/// Trait for cast f32 -> f64.
pub trait ToVecf64 {
    /// Cast f32 -> f64.
    fn to_vec_f64(&self) -> Vec<f64>;
}

impl ToVecf64 for Vec<f32> {
    fn to_vec_f64(&self) -> Vec<f64> {
        self.iter().map(|&e| e as f64).collect()
    }
}
