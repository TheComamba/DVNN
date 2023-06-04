use ndarray::{arr1, Array1};
use neuralnet::Val;

mod neuralnet;

fn main() {
    // let training_input = read_input("dataset/train-images-idx3-ubyte").unwrap();
    // let training_output = read_output("dataset/train-labels-idx1-ubyte").unwrap();
    let training_input = read_input("dataset/t10k-images-idx3-ubyte").unwrap();
    let training_output = read_output("dataset/t10k-labels-idx1-ubyte").unwrap();

    let training_dataset = training_input
        .iter()
        .zip(training_output.iter())
        .map(|(input, output)| (input.clone(), *output))
        .collect::<Vec<_>>();

    let node_numbers = vec![training_input.len(), 3, training_output.len()];
    let mut net = neuralnet::Neuralnet::new(node_numbers);
    net.train(&training_dataset);

    let test_input = read_input("dataset/t10k-images-idx3-ubyte").unwrap();
    let test_output = read_output("dataset/t10k-labels-idx1-ubyte").unwrap();
    let test_dataset = test_input
        .iter()
        .zip(test_output.iter())
        .map(|(input, output)| (input.clone(), *output))
        .collect::<Vec<_>>();
    let error = net.total_error(&test_dataset);
    println!("Error: {}", error);
}

fn read_input(path: &str) -> Result<Vec<Array1<Val>>, std::io::Error> {
    let file = std::fs::File::open(path)?;
    let decode =
        idx_decoder::IDXDecoder::<_, idx_decoder::types::U8, nalgebra::U3>::new(file).unwrap();
    let mut images = Vec::new();
    for val in decode {
        images.push(arr1(&val.iter().map(|e| *e as Val).collect::<Vec<_>>()));
        if images.len() % 1000 == 0 {
            println!("Read {} images.", images.len());
        }
    }
    Ok(images)
}

fn read_output(path: &str) -> Result<Vec<Val>, std::io::Error> {
    let file = std::fs::File::open(path)?;
    let decode =
        idx_decoder::IDXDecoder::<_, idx_decoder::types::U8, nalgebra::U1>::new(file).unwrap();
    let decode = decode.map(|e| e as Val).collect::<Vec<_>>();
    Ok(decode)
}
