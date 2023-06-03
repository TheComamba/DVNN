use ndarray::arr2;

fn main() {
    println!("Hello, world!");
}

struct Neuralnet {
    weights: Vec<arr2<u16>>,
}

impl Neuralnet {
    fn new(layer_sizes: Vec<u8>) -> Neuralnet {
        let mut weights = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            let mut layer_weights = Matrix::new(layer_sizes[i], layer_sizes[i + 1]);
            weights.push(layer_weights);
        }
        Neuralnet {
            weights: Vec::new(),
        }
    }

    fn feedforward(&self, input: Vec<u16>) -> Vec<u16> {
        let mut output = input;
        for i in 0..self.weights.len() {
            output = self.weights[i].dot(output);
            for j in 0..output.len() {
                output[j] = sign(output[j]);
            }
        }
        output
    }

    fn total_error(&self, dataset: Vec<Pair<u16, u16>>) {
        let mut error = 0;
        for i in 0..dataset.len() {
            let input = dataset[i].0;
            let target = dataset[i].1;
            let output = self.feedforward(input);
            error += (target - output) ^ 2;
        }
        return error;
    }

    const EVOLUTION_SAMPLE_SIZE: u8 = 8;
    const MAX_CHANGE: u16 = 10;

    pub fn train(&mut self, dataset: Vec<Pair<u16, u16>>) {
        for i in 0..Self::EVOLUTION_SAMPLE_SIZE {
            let mut new_weights = self.weights;
            for j in 0..self.weights.len() {
                for k in 0..self.weights[j].len() {
                    for l in 0..self.weights[j][k].len() {
                        new_weights[j][k][l] = self.weights[j][k][l]
                            + rand::random::<u16>() % 2 * Self::MAX_CHANGE
                            - Self::MAX_CHANGE;
                    }
                }
            }
            let mut new_net = Neuralnet {
                weights: new_weights,
            };
            let error = new_net.total_error(dataset);
            if error < self.total_error(dataset) {
                self.weights = new_weights;
            }
        }
    }
}
