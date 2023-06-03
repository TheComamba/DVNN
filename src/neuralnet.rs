use ndarray::{Array, Array1, Array2, ShapeBuilder};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

pub(crate) type Val = i32;

pub(crate) struct Neuralnet {
    weights: Vec<Array2<Val>>,
}

impl Neuralnet {
    pub(crate) fn new(layer_sizes: Vec<usize>) -> Neuralnet {
        let mut weights = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            let size = (layer_sizes[i], layer_sizes[i + 1]);
            let layer_weights = Array2::<Val>::zeros(size.f());
            weights.push(layer_weights);
        }
        Neuralnet { weights }
    }

    fn feedforward(&self, input: &Array1<Val>) -> Array1<Val> {
        let mut output = input.clone();
        for i in 0..self.weights.len() {
            output = self.weights[i].dot(&output);
            for j in 0..output.len() {
                output[j] = output[j].signum();
            }
        }
        output
    }

    pub(crate) fn total_error(&self, dataset: &Vec<(Array1<Val>, Val)>) -> f64 {
        let mut error = 0.0;
        for i in 0..dataset.len() {
            let input = &dataset[i].0;
            let output = self.feedforward(&input);
            let mut target = Array1::<Val>::zeros(output.len());
            target[dataset[i].1 as usize] = 1;
            println!("Target: {:?}", target.to_vec());
            println!("Output: {:?}", output.to_vec());
            let diff = target - output;
            println!("Diff: {:?}", diff.to_vec());
            error += (diff.dot(&diff) / dataset.len() as Val) as f64;
        }
        error
    }

    const MAX_STEPS: usize = 8000;
    const MAX_CHANGE: Val = 10;
    const ERROR_THRESHOLD: f64 = 10.0;

    pub fn train(&mut self, dataset: &Vec<(Array1<Val>, Val)>) {
        for i in 0..Self::MAX_STEPS {
            println!("Training step {}", i);
            let mut new_weights = Vec::new();
            for weight in self.weights.iter() {
                let height = weight.shape()[0];
                let width = weight.shape()[1];
                let delta = Array::random(
                    (height, width),
                    Uniform::new(-Self::MAX_CHANGE, Self::MAX_CHANGE),
                );
                new_weights.push(weight + delta);
            }
            let new_net = Neuralnet {
                weights: new_weights,
            };
            let current_error = self.total_error(dataset);
            let new_error = new_net.total_error(dataset);
            println!("CurrentError: {}, Newerror: {}", current_error, new_error);
            if new_error < current_error {
                self.weights = new_net.weights;
                if new_error < Self::ERROR_THRESHOLD {
                    break;
                }
            }
        }
    }
}
