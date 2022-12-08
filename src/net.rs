use autograph::{
    learn::neural_network::layer::{Conv, Dense, Forward, Layer, MaxPool, Relu},
    result::Result,
};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[derive(Clone, Copy)]
pub enum NetworkKind {
    Linear,
    CNN,
    Lenet5,
    Lenet,
}

#[derive(Layer, Forward, Clone, Debug, Serialize, Deserialize)]
pub struct CNN {
    #[autograph(layer)]
    conv1: Conv,
    #[autograph(layer)]
    relu1: Relu,
    #[autograph(layer)]
    dense1: Dense,
    #[autograph(layer)]
    relu2: Relu,
    #[autograph(layer)]
    dense2: Dense,
}

impl CNN {
    pub fn new() -> Result<Self> {
        // L0 | #: (28, 28) -> (, , )
        let conv1 = Conv::from_inputs_outputs_kernel(1, 6, [5, 5]);
        let relu1 = Relu::default();

        // L1 | #: (, , ) -> (, , )
        let dense1 = Dense::from_inputs_outputs(6 * 24 * 24, 84);
        let relu2 = Relu::default();

        // L2 | #: (, , ) -> (10)
        let dense2 = Dense::from_inputs_outputs(84, 10).with_bias(true)?;

        Ok(Self {
            conv1,
            relu1,
            dense1,
            relu2,
            dense2,
        })
    }
}

#[derive(Layer, Forward, Clone, Debug, Serialize, Deserialize)]
pub struct Lenet5 {
    #[autograph(layer)]
    conv1: Conv,
    #[autograph(layer)]
    relu1: Relu,
    #[autograph(layer)]
    pool1: MaxPool,
    #[autograph(layer)]
    conv2: Conv,
    #[autograph(layer)]
    relu2: Relu,
    #[autograph(layer)]
    pool2: MaxPool,
    #[autograph(layer)]
    dense1: Dense,
    #[autograph(layer)]
    relu3: Relu,
    #[autograph(layer)]
    dense2: Dense,
    #[autograph(layer)]
    relu4: Relu,
    #[autograph(layer)]
    dense3: Dense,
}

impl Lenet5 {
    pub fn new() -> Result<Self> {
        // L0 | #: (28, 28) -> (, , )
        let conv1 = Conv::from_inputs_outputs_kernel(1, 6, [5, 5]);
        let relu1 = Relu::default();
        let pool1 = MaxPool::from_kernel([2, 2]).with_strides(2)?;

        // L1 | #: (, , ) -> (4, 4, 16)
        let conv2 = Conv::from_inputs_outputs_kernel(6, 16, [5, 5]);
        let relu2 = Relu::default();
        let pool2 = MaxPool::from_kernel([2, 2]).with_strides(2)?;

        // L2 | #: (4, 4, 16) -> (120)
        let dense1 = Dense::from_inputs_outputs(16 * 4 * 4, 120);
        let relu3 = Relu::default();

        // L3 | #: (120) -> (84)
        let dense2 = Dense::from_inputs_outputs(120, 84);
        let relu4 = Relu::default();

        // L4 | #: (84) -> (10)
        let dense3 = Dense::from_inputs_outputs(84, 10).with_bias(true)?;

        Ok(Self {
            conv1,
            relu1,
            pool1,
            conv2,
            relu2,
            pool2,
            dense1,
            relu3,
            dense2,
            relu4,
            dense3,
        })
    }
}

#[derive(Layer, Forward, Clone, Debug, Serialize, Deserialize)]
pub struct Lenet {
    #[autograph(layer)]
    conv1: Conv,
    #[autograph(layer)]
    relu1: Relu,
    #[autograph(layer)]
    pool1: MaxPool,
    #[autograph(layer)]
    conv2: Conv,
    #[autograph(layer)]
    relu2: Relu,
    #[autograph(layer)]
    pool2: MaxPool,
    #[autograph(layer)]
    dense1: Dense,
    #[autograph(layer)]
    relu3: Relu,
    #[autograph(layer)]
    dense2: Dense,
    #[autograph(layer)]
    relu4: Relu,
    #[autograph(layer)]
    dense3: Dense,
}

impl Lenet {
    pub fn new() -> Result<Self> {
        // L0 | #: (28, 28) -> (, , )
        let conv1 = Conv::from_inputs_outputs_kernel(1, 6, [5, 5]);
        let relu1 = Relu::default();
        let pool1 = MaxPool::from_kernel([2, 2]).with_strides(2)?;

        // L1 | #: (, , ) -> (, , )
        let conv2 = Conv::from_inputs_outputs_kernel(6, 16, [5, 5]);
        let relu2 = Relu::default();
        let pool2 = MaxPool::from_kernel([2, 2]).with_strides(2)?;

        // L2 | #: (, , ) -> (, , )
        //let conv3 = Conv::from_inputs_outputs_kernel(16, #, [4, 4]);
        //let relu3 = Relu::default();
        //let pool3 = MaxPool::from_kernel([2, 2]).with_strides(2)?;

        // L3 | #: (, , ) -> (120)
        let dense1 = Dense::from_inputs_outputs(16 * 4 * 4, 120);
        let relu3 = Relu::default();

        // L4 | #: (120) -> (84)
        let dense2 = Dense::from_inputs_outputs(120, 84);
        let relu4 = Relu::default();

        // L5 | #: (84) -> (10)
        let dense3 = Dense::from_inputs_outputs(84, 10).with_bias(true)?;

        Ok(Self {
            conv1,
            relu1,
            pool1,
            conv2,
            relu2,
            pool2,
            dense1,
            relu3,
            dense2,
            relu4,
            dense3,
        })
    }
}
