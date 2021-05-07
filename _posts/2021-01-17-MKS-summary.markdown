---
layout: post
title:  "Looking for Structure-Property Linkages"
date:   2021-06-17 6:30:00 -0500
categories: introduction
---

# Why does progress in materials discovery move so slowly?
We live in a world we progress is expected to happen at an increasingly rapid pace. New products are released each and every year that promise bigger, faster, more efficient, etc. But the same rapid progress hasn't necessarily been apparent in the materials development community. Often, it can take roughly 20 years for a new material discovery to hit the market ([MGI](https://www.mgi.gov)). One of the main contributions to this lack of progress is because the design space for developing new materials is infinite. Take steel for example - changing the amount of Carbon and Iron in the material, changing the material processing history, will alter the final performance of the resulting material. So its super difficult to explore all of the possible combinations of variables in a material discovery problem to come up with the optimal combination for the desired problem.

One of the fundamental insights we have gained is that a material's manufacturing **process** dictates the internal **structure** of the material, which in turn determines the material's **performance**. Remember that flow: Process -> Structure -> Property. (We'll focus on the structure <-> property link in this post). Since we know that a material's structure will determine the final performance, a lot of effort has been spent in figuring out the physics that dictate the relationship. In other words, we want to find structure-property linkages. If we can find out how the two are related, we could check how a material might behave given its internal structure, or we could invert the problem and start with a desired performance characteristic and try and find the structure that would produce that property.

> Think of how a house is built. Inside of the walls are studs that add structural rigidity and support. So the structure is 2x4's and the performance is support in holding the roof up.

So we know what we need to do (find the structure-property linkage). What's taking so long then? Well, like I said earlier... The design space is enormous. It takes a long time to check all of the possible structure candidates to arrive at a property candidate that we want. To add to this issue, the ways that we evaluate the structure are slow and expensive. To simplify things, lets look at this diagram:

![slow](/assets/slow_material_development.jpg)

Traditional methods have used finite element analysis (FEA) or experimentation to learn the structure-property linkage and evaluate new materials. Note that this process is repetitive - we try a new microstructure, check the results, then try a new one. Again and again. Again, since the number of possible microstructures is massive, we have to check a lot. The main issue here is that experiments are slow and expensive and FEA is way too slow for solving these problems computationally. So where I find myself now is working on new ways to speed up this process of finding structure-property linkages and evaluating candidate microstructures. This is done through a surrogate model.
> By surrogate, I mean that we use machine learning and data science approaches to train a model that replaces the "traditional" approaches. 

Our new workflow with a surrogate model looks something like this:

![fast](/assets/fast.jpg)

The advantage comes in that we can train our surrogate a single time on existing data and then use the surrogate to analyze new microstructures. Another advantage is that the surrogate can "learn" the structure-property link that we've been wanting. Carefully choosing and formulating our surrogate can lead to massive improvements in the time it takes to analyze new microstructures. In the next section, I'll touch on one of the ways my lab has formulated a surrogate model for this problem.

# In comes MKS!

It turns out there are numerous data science and machine learning approaches to solving these sorts of problems. I'm going to talk about one in particle called the Materials Knowledge System [(MKS)](http://pymks.org/en/stable/rst/README.html). The basic principle of the MKS framework is that it efficiently learns structure-property linkages through calibration on an existing dataset. The weights are calibrated using [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression) and once this task is complete, the linkages can be used to analyze the performance of any microstructure (As long as you stay within the same material system). The final form of MKS is:

$$ p_{ij,\textbf{x}} = \bar{p}_{ij} + \sum_{\sigma, h} \alpha^h_{ij,\sigma} m^h_{\textbf{x}-\sigma} + \sum_{\sigma, \sigma', h, h'}\alpha^{h,h'}_{ij,\sigma, \sigma'}m^h_{\textbf{x}-\sigma}m^{h'}_{\textbf{x}-\sigma'}+ . . . $$

> $p_{ij}$ can be any material property of interest (in our case, we picked strain: $\varepsilon$).\
$m^h$ is what we call the "microstructure function" and its job is to indicate which "state" the material is in.\
$\alpha^h_{ij,\sigma}$ is how we represent those structure-property linkages I mentioned earlier. Notice how it "operates" on the microstructure function (structure) and produces the strain (property).\
Lastly, if you've never seen subscripts or superscripts before then check out [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation)


## Beginning our derivation

Like most things in physics and engineering, we'll have to start this whole process with a differential equation. Specifically this one: 

$$\sigma_{ij,j} = 0$$

This equation is simply a statement of [Conservation of Linear Momentum](https://en.wikipedia.org/wiki/Momentum#Conservation_in_a_continuum) at equilibrium in the absence of body forces (body force is gravity, magnetism, etc). To clarify, the equation describes the motion of each point within the material's body. There is some complex math behind it, but all you need to take away is that the equation states that our material must obey the condition that it is internally "continuous" - think it doesn't separate or have internal voids. That's our governing equation.

Now, $\sigma$ is the stress in our material. The issue with stress is that it doesn't describe the movement of the material's particles (it's a statement about force). Luckily we have Hooke's Law ($\sigma_{ij} = C_{ijkl} \varepsilon_{kl}$) which relates stress in a material to strain. Combining these equations we get:

$$\sigma_{ij,j} = (C_{ijkl}\varepsilon_{kl})_{,j} = 0$$

> Note that $C_{ijkl}$ is our stiffness tensor - it determines how much our material "resists" bending or stretching 

Which is convenient because now we have a governing differential equation that describes motion. But now that we have our differential equation ... how do we solve it? It turns out that you can manipulate the equation again so that it's in a form where we can use some pretty neat tricks to help us solve the equatoin. The manipulation is built on [perturbation theory](https://en.wikipedia.org/wiki/Perturbation_theory) - we will split our stiffness tensor and strain into a "reference" part and a "perturbed" part:

$$C_{ijkl}(x) = C^R_{ijkl} + C'_{ijkl}(x) $$
$$\varepsilon_{ij}(x) = \varepsilon^R_{ij} + \varepsilon'_{ij}(x)$$

With some work, we can now get our differential equation to look like this:

$$C^R_{ijkl}\varepsilon'_{kl,j}(x) = F_i(x)$$

Where 

$$F_i(x) = -[C'_{ijkl}(x)\varepsilon_{kl}(x)]_{,j}$$

Now our equation is an **inhomogeneous**, **constant coefficient**, **linear**, **elliptic** partial differential equation. This form is helpful because we can use a [Green's function](https://en.wikipedia.org/wiki/Green%27s_function) to solve it. I encourage you to dig deeper into what Green's functions do - it's incredible what techniques were developed so many years ago to help solve these problems without computers. Maybe later I'll write a post solely on Green's functions, but for now, we'll introduce $G_{ijk}(x-s)$ as our Green's function. Utilizing our Green's function (and doing some minor manipulations), our equation now becomes:

$$\varepsilon_{ij}(x) = \varepsilon^R_{ij}(x) - \int_v G_{ijk,l}(x-s)C'_{klmn}(s)\varepsilon_{mn}(s)d^3s$$

Which is kind of cool. Though we haven't really solved anything, we've manipulated our equations into increasingly useful forms. Before moving forward, let's remind ourselves what we are really trying to do here. We want to figure out the mathematical relationship between a material's **structure** and its **property**. Up to this point, we just have an equation with a Green's function that we don't know and strain shows up on both sides. It doesn't seem super helpful. So let's try a few more things. First, we'll convert this to "discrete space". Discrete space is the world of computers.

$$\varepsilon_{ij,x} = \varepsilon^R_{ij,x} - \sum_s \Gamma_{ijkl,x-s}C'_{klmn,s}\varepsilon_{mn,s}\Delta V$$

> $\Gamma_{ijkl} = G_{ijk,l}$ will help simplify things\
$\Delta V$ is the discrete volume of each "cell" in our material

