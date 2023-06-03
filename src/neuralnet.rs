use ndarray::{Array, Array1, Array2, ShapeBuilder};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

pub(crate) struct Neuralnet {
    weights: Vec<Array2<i16>>,
}

impl Neuralnet {
    pub(crate) fn new(layer_sizes: Vec<usize>) -> Neuralnet {
        let mut weights = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            let size = (layer_sizes[i], layer_sizes[i + 1]);
            let layer_weights = Array2::<i16>::zeros(size.f());
            weights.push(layer_weights);
        }
        Neuralnet {
            weights: Vec::new(),
        }
    }

    fn feedforward(&self, input: &Array1<i16>) -> Array1<i16> {
        let mut output = input.clone();
        for i in 0..self.weights.len() {
            output = self.weights[i].dot(&output);
            for j in 0..output.len() {
                output[j] = output[j].signum();
            }
        }
        output
    }

    pub(crate) fn total_error(&self, dataset: &Vec<(Array1<i16>, i16)>) -> i16 {
        let mut error = 0;
        for i in 0..dataset.len() {
            let input = &dataset[i].0;
            let output = self.feedforward(&input);
            let mut target = Array1::<i16>::zeros(output.len());
            target[dataset[i].1 as usize] = 1;
            let diff = target - output;
            error += diff.dot(&diff);
        }
        error
    }

    const MAX_STEPS: usize = 8;
    const MAX_CHANGE: i16 = 10;
    const ERROR_THRESHOLD: i16 = 10;

    pub fn train(&mut self, dataset: &Vec<(Array1<i16>, i16)>) {
        for _ in 0..Self::MAX_STEPS {
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
            let new_error = new_net.total_error(dataset);
            let current_error = self.total_error(dataset);
            if new_error < current_error {
                self.weights = new_net.weights;
                if new_error < Self::ERROR_THRESHOLD {
                    break;
                }
            }
        }
    }
}
