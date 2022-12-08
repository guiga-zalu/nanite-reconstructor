# Nanite Reconstructor

## Conceptuality

Inspired by "[Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)", 
as [showcased](https://www.youtube.com/watch?v=bXzauli1TyU) by [Two Minute Papers](https://www.youtube.com/@TwoMinutePapers), 
I thought of a pathway to implement these image regeneration capabilities in generic scenarios.

This will be achieved by using a CNN ("Convolutional Neural Networks") trained to as how "Diffusion models" are. 
It will be run iteratively, where:

- For each iteration:
  - For each pixel:
    1. If the pixel is already there, skip
    1. If at least $\alpha \%$ of the neighbouring pixels are there,
       1. run the network with these pixels as input
       1. sets the newly generated pixel

The training will be done as follows:

1. An input image ($\# I = \left< \mathrm{width}, \mathrm{height}, \mathrm{channels} \right>$) is selected from \[a whole image of a\] sample set
1. An input boolean mask ($\# M = \left< \mathrm{width}, \mathrm{height} \right>$) is generated, representing the lack of pixels
1. The net is trained to, for given local $I$ and $M$ (the size of the CNNs input kernel), reproduce every pixel $p$ where the mask $m$ is `false`.
   The network won't have access to any pixel where $m$ is `false`.
1. Remove pixels from the mask $M$ (by setting them to `false`) before resuming training

### Utilities

Depending on the capabilities of abstraction (relative to the model) and on the variety of the training, 
this network could be used to (in the context of images):

- Remove noise, given a explicit pixel-wise definition of the noise.
- Remove elements with complete regeneration
- Extend edges
- Connect multiple images, or even image streams (videos):
  In the subcontext of webcam chat, instead of having borders or removing the background and "placing" the people in "chairs" (as is done by Skype), fully extend their backgrounds and create one full room, where "everyone" "is present"

And with the concepts of "Transfer Learning", a pre-trained network could be specialized into an specific builder or reconstructor.

### Math

The initial image is $I_0$, with it's binary mask $M_0$. In this mask, a `true` value is the presence of a pixel, while a `false` value is the absence of it.  
The subsequent images and masks follow the iterative rule $(I, M)_{n + 1} = f\circ(I, M)_n$.

$$
\newcommand{\false}{\mathrm{false}}
\newcommand{\true}{\mathrm{true}}
\begin{aligned}
    B\circ(M, \vec p, \vec k) &= \dfrac{
        \sum_{\vec i\in\pm\vec k}
            M\circ (\vec p + \vec i)
    }{
        \#\{
            (\vec p + \vec i)\in M \forall\vec i\in\pm\vec k
        \}
    }\\
    f\circ(I, M, \vec p) &= \begin{cases}
        M\circ\vec p = \true: &(I\circ\vec p, \true)\\
        B\circ(\cdot) < \alpha: &(0, \false)\\
        &(g\circ[I, M, \vec p, \vec k], \true)
    \end{cases}
\end{aligned}
$$

Unknown $f$ params:

- $\vec k\in\mathrm N^n$: the kernel shape minus $\vec 1$ over $2$
- $\alpha\in[0; 1]$

While $g\circ(I, M, \vec p, \vec k)$ is a descending series of convolutions on the neighbourhood of $\vec p$, resulting in only one pixel.
