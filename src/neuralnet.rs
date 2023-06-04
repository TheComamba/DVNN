use ndarray::{Array, Array1, Array2};
use ndarray_rand::RandomExt;
use rand::{distributions::Uniform, prelude::Distribution};

pub(crate) type Val = i32;

pub(crate) struct Neuralnet {
    weights: Vec<Array2<Val>>,
}

impl Neuralnet {
    pub(crate) fn new(layer_sizes: Vec<usize>) -> Neuralnet {
        let mut weights = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            let size = (layer_sizes[i + 1], layer_sizes[i]);
            let layer_weights = Array2::<Val>::zeros(size);
            weights.push(layer_weights);
        }
        Neuralnet { weights }
    }

    fn make_step(vec: &mut Array1<Val>) {
        for j in 0..vec.len() {
            vec[j] = vec[j].signum();
        }
    }

    fn normalise_to_percent(vec: &mut Array1<Val>) {
        for j in 0..vec.len() {
            if vec[j] < 0 {
                vec[j] = 0;
            }
        }
        let mut sum = 0;
        for j in 0..vec.len() {
            sum += vec[j];
        }
        if sum == 0 {
            return;
        }
        for j in 0..vec.len() {
            vec[j] = vec[j] * 100 / sum;
        }
    }

    fn feedforward(&self, input: &Array1<Val>) -> Array1<Val> {
        let mut output = input.clone();
        for i in 0..self.weights.len() {
            output = self.weights[i].dot(&output);
            if i != self.weights.len() - 1 {
                Self::make_step(&mut output);
            } else {
                Self::normalise_to_percent(&mut output);
            }
        }
        output
    }

    const PERCENT_SQUARED: f64 = (100 * 100) as f64;

    pub(crate) fn total_error(&self, dataset: &Vec<(Array1<Val>, Val)>) -> f64 {
        let mut error = 0.0;
        for i in 0..dataset.len() {
            let input = &dataset[i].0;
            let output = self.feedforward(&input);
            for j in 0..output.len() {
                let target = if j == dataset[i].1 as usize { 100 } else { 0 };
                error += (output[j] - target).pow(2) as f64;
            }
        }
        error as f64 / dataset.len() as f64 / Self::PERCENT_SQUARED
    }
    const MAX_CHANGE: Val = 10;
    const ERROR_THRESHOLD: f64 = 0.05;

    pub fn train(&mut self, dataset: &Vec<(Array1<Val>, Val)>) {
        let mut i = 0;
        loop {
            let mut new_weights = Vec::new();
            for weight in self.weights.iter() {
                let height = weight.shape()[0];
                let width = weight.shape()[1];
                let change_dist = Uniform::new(1, Self::MAX_CHANGE).sample(&mut rand::thread_rng());
                let delta = Array::random((height, width), Uniform::new(-change_dist, change_dist));
                new_weights.push(weight + delta);
            }
            let new_net = Neuralnet {
                weights: new_weights,
            };
            let current_error = self.total_error(dataset);
            let new_error = new_net.total_error(dataset);
            if new_error < current_error {
                println!(
                    "Step {}: FormerError = {}, Newerror = {}",
                    i, current_error, new_error
                );
                self.weights = new_net.weights;
                if new_error < Self::ERROR_THRESHOLD {
                    break;
                }
            }
            i += 1;
        }
    }
}
