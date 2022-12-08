use argparse::{ArgumentParser, Store, StoreConst, StoreTrue};
use autograph::{device::Device, learn::neural_network::layer::Dense, result::Result};
use std::path::PathBuf;

use nanite_reconstructor::{
    net::{Lenet, Lenet5, NetworkKind, CNN},
    train,
};

#[tokio::main]
async fn main() -> Result<()> {
    let mut kind = NetworkKind::Linear;
    let mut save = false;
    let mut epochs = 100;
    {
        let mut ap = ArgumentParser::new();
        ap.set_description("Neural Network MNIST Example");
        ap.refer(&mut kind)
            .add_option(
                &["--linear"],
                StoreConst(NetworkKind::Linear),
                "A linear network.",
            )
            .add_option(
                &["--cnn"],
                StoreConst(NetworkKind::CNN),
                "A convolutional network.",
            )
            .add_option(
                &["--lenet5"],
                StoreConst(NetworkKind::Lenet5),
                "The LeNet5 network.",
            )
            .add_option(
                &["--lenet"],
                StoreConst(NetworkKind::Lenet),
                "My LeNet network!",
            );
        ap.refer(&mut save)
            .add_option(&["--save"], StoreTrue, "Load / save the trainer.");
        ap.refer(&mut epochs).add_option(
            &["--epochs"],
            Store,
            "The number of epochs to train for.",
        );
        ap.parse_args_or_exit();
    }

    println!("Arguments: {save} {epochs}");
    // Create a device.
    let device = Device::new()?;
    println!("{:#?}", device);

    match kind {
        NetworkKind::Linear => {
            let dense = Dense::from_inputs_outputs(28 * 28, 10).with_bias(true)?;
            train(device, dense, save_path("linear", save), epochs).await
        }
        NetworkKind::CNN => train(device, CNN::new()?, save_path("cnn", save), epochs).await,
        NetworkKind::Lenet5 => {
            train(device, Lenet5::new()?, save_path("lenet5", save), epochs).await
        }
        NetworkKind::Lenet => train(device, Lenet::new()?, save_path("lenet", save), epochs).await,
    }
}

fn save_path(name: &str, save: bool) -> Option<PathBuf> {
    if save {
        Some(format!("{}_trainer.bincode", name).into())
    } else {
        None
    }
}
