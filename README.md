# Patterns of Nature
This is a repository for storing things that generate patterns of nature. Currently implemented:

* Branching growth (hyphae.py)

## Hyphae

This is based on a model of plant vein growth due to sources of auxins that pop in and out of existence. The implementation is [based on this paper](http://algorithmicbotany.org/papers/venation.sig2005.pdf) from the University of Calgary's computational botany group. The main difference between my implementation and that one is that I do nearest neighbor calculations using KD-Trees instead of Voronoi diagrams and that my parameters vary with time.

https://user-images.githubusercontent.com/LumenPallidium/morphogenesis/raw/main/images/hyphae.mp4

I drew aesthetic inspiration from [this repository](https://github.com/jblondin/hyphae/tree/master).

# Further Reading

[This repository is wonderful.](https://github.com/jasonwebb/morphogenesis-resources)
