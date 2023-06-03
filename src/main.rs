use ndarray::{arr1, Array1};

mod neuralnet;

fn main() {
    let training_input = read_input("dataset/train-images.idx3-ubyte").unwrap();
    let training_output = read_output("dataset/train-labels.idx1-ubyte").unwrap();
    let training_dataset = training_input
        .iter()
        .zip(training_output.iter())
        .map(|(input, output)| (input.clone(), *output))
        .collect::<Vec<_>>();

    let node_numbers = vec![training_input.len(), 16, 16, training_output.len()];
    let mut net = neuralnet::Neuralnet::new(node_numbers);
    net.train(&training_dataset);

    let test_input = read_input("dataset/t10k-images.idx3-ubyte").unwrap();
    let test_output = read_output("dataset/t10k-labels.idx1-ubyte").unwrap();
    let test_dataset = test_input
        .iter()
        .zip(test_output.iter())
        .map(|(input, output)| (input.clone(), *output))
        .collect::<Vec<_>>();
    let error = net.total_error(&test_dataset);
    println!("Error: {}", error);
}

fn read_input(path: &str) -> Result<Vec<Array1<i16>>, std::io::Error> {
    let file = std::fs::File::open(path)?;
    let decode =
        idx_decoder::IDXDecoder::<_, idx_decoder::types::I16, nalgebra::U1>::new(file).unwrap();
    let mut input = Vec::new();
    let mut image = Vec::new();
    let mut size = 0;
    for val in decode {
        image.push(val);
        size += 1;
        if size == 28 * 28 {
            input.push(arr1(&image));
            image.clear();
            size = 0;
        }
    }
    Ok(input)
}

fn read_output(path: &str) -> Result<Vec<i16>, std::io::Error> {
    let file = std::fs::File::open(path)?;
    let decode =
        idx_decoder::IDXDecoder::<_, idx_decoder::types::I16, nalgebra::U1>::new(file).unwrap();
    let decode = decode.collect::<Vec<_>>();
    Ok(decode)
}
