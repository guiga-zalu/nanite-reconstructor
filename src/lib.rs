#![feature(type_alias_impl_trait)]

use autograph::{
    dataset::mnist::Mnist,
    device::Device,
    learn::{
        neural_network::{layer::Layer, NetworkTrainer},
        Summarize, Test, Train,
    },
    result::Result,
    tensor::{float::FloatTensor4, Tensor, Tensor1, TensorView},
};
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use ndarray::{s, Array1, ArrayView1, ArrayView4, Axis};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, fs, path::PathBuf};

pub mod net;

pub async fn train<L>(
    device: Device,
    layer: L,
    save_path: Option<PathBuf>,
    epochs: usize,
) -> Result<()>
where
    L: Layer + Debug + Serialize + for<'de> Deserialize<'de>,
{
    // Construct a trainer to train the network.
    let mut trainer: NetworkTrainer<L> = match save_path.as_ref() {
        // Load the trainer from a file.
        Some(save_path) if save_path.exists() => bincode::deserialize(&fs::read(save_path)?)?,
        // Use the provided layer.
        _ => NetworkTrainer::from_network(layer.into()),
    };

    // Transfer the trainer to the device. Most operations (ie compute shaders) only
    // run on a device.
    trainer.to_device_mut(device.clone()).await?;
    println!("{:#?}", &trainer);

    // Load the dataset.
    println!("Loading dataset...");
    let mnist = Mnist::builder().download(true).build()?;
    // Use the first 60_000 images as the training set.
    let train_images = mnist.images().slice(s![..60_000, .., .., ..]);
    let train_classes = mnist.classes().slice(s![..60_000]);
    let train_batch_size = 100;
    // Use the last 10_000 images as the test set.
    let test_images = mnist.images().slice(s![60_000.., .., .., ..]);
    let test_classes = mnist.classes().slice(s![60_000..]);
    let test_batch_size = 1000;

    println!("Training...");
    // Run the training for the specified epochs
    let mut epoch = trainer.summarize().epoch;
    while epoch < epochs {
        let (train_iter, test_iter) = get_iters(
            &device,
            &train_images,
            &train_classes,
            train_batch_size,
            &test_images,
            &test_classes,
            test_batch_size,
            epoch,
        );
        trainer.train_test(train_iter, test_iter)?;
        let summary = trainer.summarize();
        epoch = summary.epoch;
        println!("{:#?}", summary);

        save_trainer(&save_path, &trainer)?;
    }

    println!("Evaluating...");
    let test_iter = progress_iter(
        batch_iter(&device, &test_images, &test_classes, test_batch_size),
        trainer.summarize().epoch,
        "evaluating",
    );
    let stats = trainer.test(test_iter)?;
    println!("{:#?}", stats);

    Ok(())
}

type IteratorItem = Result<(FloatTensor4, Tensor1<u8>)>;
type TrainIterator<'a> = impl ExactSizeIterator<Item = IteratorItem> + 'a;
type TestIterator<'a> = impl ExactSizeIterator<Item = IteratorItem> + 'a;

fn get_iters<'a>(
    device: &'a Device,
    train_images: &'a ArrayView4<'a, u8>,
    train_classes: &'a ArrayView1<'a, u8>,
    train_batch_size: usize,
    test_images: &'a ArrayView4<'a, u8>,
    test_classes: &'a ArrayView1<'a, u8>,
    test_batch_size: usize,
    epoch: usize,
) -> (TrainIterator<'a>, TestIterator<'a>) {
    let train_iter = progress_iter(
        shuffled_batch_iter(device, &train_images, &train_classes, train_batch_size),
        epoch,
        "training",
    );
    let test_iter = progress_iter(
        batch_iter(&device, &test_images, &test_classes, test_batch_size),
        epoch,
        "testing",
    );
    (train_iter, test_iter)
}

fn save_trainer<L>(save_path: &Option<PathBuf>, trainer: &NetworkTrainer<L>) -> Result<()>
where
    L: Layer + Debug + Serialize + for<'de> Deserialize<'de>,
{
    if let Some(save_path) = save_path.as_ref() {
        fs::write(save_path, bincode::serialize(trainer)?)?;
    }
    Ok(())
}

// Stream the data to the device, converting arrays to tensors.
fn batch_iter<'a>(
    device: &'a Device,
    images: &'a ArrayView4<u8>,
    classes: &'a ArrayView1<u8>,
    batch_size: usize,
) -> impl ExactSizeIterator<Item = IteratorItem> + 'a {
    images
        .axis_chunks_iter(Axis(0), batch_size)
        .into_iter()
        .zip(classes.axis_chunks_iter(Axis(0), batch_size))
        .map(move |(x, t)| {
            let x = smol::block_on(TensorView::try_from(x)?.into_device(device.clone()))?
                // normalize the bytes to f32
                .scale_into::<f32>(1. / 255.)?
                .into_float();
            let t = smol::block_on(TensorView::try_from(t)?.into_device(device.clone()))?;
            Ok((x, t))
        })
}

// Shuffled training data iterator
fn shuffled_batch_iter<'a>(
    device: &'a Device,
    images: &'a ArrayView4<'a, u8>,
    classes: &'a ArrayView1<'a, u8>,
    batch_size: usize,
) -> impl ExactSizeIterator<Item = IteratorItem> + 'a {
    let mut indices = (0..images.shape()[0]).into_iter().collect::<Vec<usize>>();
    indices.shuffle(&mut rand::thread_rng());
    (0..indices.len())
        .into_iter()
        .step_by(batch_size)
        .map(move |index| {
            let batch_indices = &indices[index..(index + batch_size).min(indices.len())];
            let shape = [
                batch_indices.len(),
                images.dim().1,
                images.dim().2,
                images.dim().3,
            ];
            let x = batch_indices
                .iter()
                .copied()
                .flat_map(|i| images.index_axis(Axis(0), i))
                .copied()
                .collect::<Array1<u8>>()
                .into_shape(shape)?;
            let t = batch_indices
                .iter()
                .copied()
                .map(|i| classes[i])
                .collect::<Array1<u8>>();
            let x = smol::block_on(Tensor::from(x).into_device(device.clone()))?
                // normalize the bytes to f32
                .scale_into::<f32>(1. / 255.)?
                .into_float();
            let t = smol::block_on(Tensor::from(t).into_device(device.clone()))?;
            Ok((x, t))
        })
}

// Show a progress bar
fn progress_iter<X>(
    iter: impl ExactSizeIterator<Item = X>,
    epoch: usize,
    name: &str,
) -> impl ExactSizeIterator<Item = X> {
    let style = ProgressStyle::default_bar()
        .template(&format!(
            "[epoch: {} elapsed: {{elapsed}}] {} [{{bar}}] {{pos:>7}}/{{len:7}} [eta: {{eta}}]",
            epoch, name
        ))
        .progress_chars("=> ");
    let bar = ProgressBar::new(iter.len() as u64).with_style(style);
    iter.progress_with(bar)
}
