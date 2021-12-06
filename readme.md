# Simple Probabilistic Programs
Simple implementation of probabilistic programs for the Julia programming language with support for traces allocated on the stack.

## Disclaimer
This package is very much work in progress and it is likely to change in the future. Comments, suggestions, and pull requests are welcome!

## Quickstart
```julia-repl
julia> using SimpleProbabilisticPrograms

julia> using Distributions: Beta, Bernoulli

julia> @probprog function beta_bernoulli_model(a, b, n)
         bias  ~ Beta(a, b)      # sample a number between 0 and 1
         coin  = Bernoulli(bias) # create biased coin as Bernoulli distribution
         trows ~ iid(coin, n)    # throw the coin n times
         return                  # return nothing by convention
       end
beta_bernoulli_model (generic function with 1 method)

julia> model = beta_bernoulli_model(3, 4, 10)
ProbProg(...)

julia> trace = rand(model)
(bias = 0.15035879436791896, coins = Bool[0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

julia> logpdf(model, trace)
-3.5451416292361504
```

## Introduction and Motivation
Probabilistic programming is a great way of making the power of Bayesian statistics more accessible by drawing from research in programming languages.
A probabilistic program represents a distribution over the random choices that are made in a standard execution of the program. The collection of all such choices is called a *trace* of the probabilistic program.
While probabilistic programming systems like [Gen.jl](https://github.com/probcomp/Gen.jl) and [Turing.jl](https://github.com/TuringLang/Turing.jl) (also [Pyro](https://github.com/pyro-ppl/pyro) and [PyMC3](https://github.com/pymc-devs/pymc) in python) focus on the implementation and integration of inference methods, the motivation of this package is to provide a minimal implementation of probabilistic programs with a simple API, full compositionality, and fast performance through type stability. 
This can be helpful, for example, for applications that go beyond the use cases of established probabilistic programming systems, if such systems seem to be too complex for your use case, or if you want to learn/teach how probabilistic programming works.
In particular, if you want to write down a distribution that is just a bit more complicated than the ones provided by `Distributions.jl`, this package might help you.

Implementing an extensive library of inference methods that work out of the box is not the goal of this package.

## High-Level API
This package builds on the fact that all you need to get started with using probabilistic programs are 3 things:
1. a way to define them
2. a function for drawing a trace randomly, and
3. a function for evaluating the log probability of a trace.

The specification of this high-level API is:
1. Macro `@probprog function_definition`: Define functions that construct probabilistic programs. This macro transforms sample statements indicated by the binary operator `~` into calls to a more low-level `sample` method (see the source code). Call the function generated by this macro to create a probabilistic program object (see example in Quickstart above).
2. Function `rand([rng,] prog)`: draw a random trace
3. Function `logpdf(prog, trace)`: evaluate log probability (density) of a trace