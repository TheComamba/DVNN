use ndarray::{Array, Array1, Array2, ShapeBuilder};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

fn main() {
    println!("Hello, world!");
}

struct Neuralnet {
    weights: Vec<Array2<i16>>,
}

impl Neuralnet {
    fn new(layer_sizes: Vec<usize>) -> Neuralnet {
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

    fn total_error(&self, dataset: &Vec<(Array1<i16>, Array1<i16>)>) -> i16 {
        let mut error = 0;
        for i in 0..dataset.len() {
            let input = &dataset[i].0;
            let target = &dataset[i].1;
            let output = self.feedforward(&input);
            let diff = target - output;
            error += diff.dot(&diff);
        }
        error
    }

    const EVOLUTION_SAMPLE_SIZE: usize = 8;
    const MAX_CHANGE: i16 = 10;

    pub fn train(&mut self, dataset: &Vec<(Array1<i16>, Array1<i16>)>) {
        for i in 0..Self::EVOLUTION_SAMPLE_SIZE {
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
            let error = new_net.total_error(dataset);
            if error < self.total_error(dataset) {
                self.weights = new_net.weights;
            }
        }
    }
}
