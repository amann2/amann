---
layout: post
title:  "Welcome to my personal website!"
date:   2021-01-17 6:30:00 -0500
categories: introduction
---

# Why does progress in materials discovery move so slowly?


## In comes MKS....

It turns out there are numerous data science approaches to solving these sorts of problems. I'm going to talk about one in particle called the Materials Knowledge System [(MKS)](http://pymks.org/en/stable/rst/README.html). The basic principle of the MKS framework is that it efficiently captures structure-property linkages in the form of a matrix of weights. The weights are calibrated using Linear Regression (I'll get to that later). The final form of MKS is:

$$ p_{ij,\textbf{x}} = \bar{p}_{ij} + \sum_{\sigma, h} \alpha^h_{ij,\sigma} m^h_{\textbf{x}-\sigma} + \sum_{\sigma, \sigma', h, h'}\alpha^{h,h'}_{ij,\sigma, \sigma'}m^h_{\textbf{x}-\sigma}m^{h'}_{\textbf{x}-\sigma'}+ . . . $$

$$p_{ij}$$ can be any material property of interest (most commonly it is strain $$\varepsilon$$). $$m^h$$ is what we call the "microstructure function" and its job is to indicate which "state" the material is in. I'll also descibe the subscripts in more detail later on. Don't worry if this is all very confusing right now - I will hopefully make it easier to understand throughout this article.


what else