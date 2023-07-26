use crate::net::neuralnet;
use ndarray::{arr1, Array1};

use super::neuralnet::Val;

pub(crate) const NUM_OF_PIXELS: usize = 28 * 28;
pub(crate) const NUM_OF_DIGITS: usize = 10;

pub(crate) type Dataset = Vec<(Array1<Val>, Val)>;

pub(crate) fn read_dataset(input_path: &str, output_path: &str) -> Dataset {
    let input = read_input(input_path).unwrap();
    let output = read_output(output_path).unwrap();

    input
        .iter()
        .zip(output.iter())
        .map(|(input, output)| (input.clone(), *output))
        .collect::<Vec<_>>()
}

fn load_and_train() {
    let training_dataset = read_dataset(
        "dataset/train-images-idx3-ubyte",
        "dataset/train-labels-idx1-ubyte",
    );

    let node_numbers = vec![NUM_OF_PIXELS, 16, 16, NUM_OF_DIGITS];
    let mut net = neuralnet::Neuralnet::new(node_numbers);
    net.train(&training_dataset);

    let test_dataset = read_dataset(
        "dataset/t10k-images-idx3-ubyte",
        "dataset/t10k-labels-idx1-ubyte",
    );
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
