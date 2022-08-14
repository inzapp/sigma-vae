# Sigma VAE [**(Paper)**](https://arxiv.org/pdf/2006.13202.pdf)

Sigma VAE is VAE model that automatically balance between reconstruction loss and kl divergence via variance learning

Vanilla VAE requires balancing hyperparameters to balance reconstruction loss and kl divergence

And the value is generally known to be around 0.001,

But it should be adjusted according to the input size or the latent dimension value

In addition, a vanilla VAE model may not be trained well without the kl burn technique, even though the balancing hyperparameter is used

Adjusting these hyperparameters every time to train a VAE was very terrible and this was completely solved by using the Sigma VAE

As Sigma VAE learns the variance of data directly, it ensures stable training without the need for these hyperparameters

## Optimal sigma

Even more surprisingly, you can use simple estimates of variance in data to achieve stable learning without learning variance in data

And it's very easy to implement. and it works just as well as learning to variance data directly.

<img src="https://user-images.githubusercontent.com/43339281/184525879-93ac7015-23a1-4196-8e30-6b181d3918f8.png" width="500px">

<img src="https://user-images.githubusercontent.com/43339281/184526129-a7528d75-449b-46f3-b3ba-304831660dc8.png" width="400px">

## Image quality compare to other VAE

<img src="https://user-images.githubusercontent.com/43339281/184526262-7e5f22f5-e5a9-4d0b-9227-12f0a6790510.png" width="800px">

<img src="https://user-images.githubusercontent.com/43339281/184526017-4dbce104-4e31-473b-9075-8f8b0d50ac96.png" width="800px">

<img src="https://user-images.githubusercontent.com/43339281/184526181-7a3a5102-eeff-4267-a7d1-612095a2cb74.png" width="800px">
