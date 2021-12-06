# Suppose you have a die and you suspect that it shows even numbers more 
# frequently than odd numbers. To investigate whether this is true, 
# you throw the die 1000 times.
# This tutorial explains how you can quantify how much more plausible it is that
# the die is indeed biased than that it's just a simple old unbiased die.

using SimpleProbabilisticPrograms, Distributions
const SPP = SimpleProbabilisticPrograms

normalize(xs) = xs ./ sum(xs)

# outside the model function to reduce memory allocation
const odd_die_numbers = Set([1, 3, 5])
const even_die_numbers = Set([2, 4, 6])

@probprog function biased_die(bias)
  is_even ~ Bernoulli(bias)
  possible_numbers = is_even ? even_die_numbers : odd_die_numbers
  number ~ UniformCategorical(possible_numbers)
  return
end

import SimpleProbabilisticPrograms: fromtrace, totrace
fromtrace(::typeof(biased_die), trace) = trace.number
totrace(::typeof(biased_die), number) = (is_even=iseven(number), number=number)

# check model
sum(iseven.(rand(iid(biased_die(0.7), 1000)))) / 1000

prog = biased_die(rand())
x = rand(prog)
@time logpdf(prog, x)
@time logpdf(prog, x)

@probprog function multiple_rolls(n)
  bias ~ Uniform(0, 1)
  numbers ~ iid(biased_die(bias), n)
  return
end

import Random: default_rng

# symmetric metrolopis hastings step
function mhsym(trace, prog, prop, get, set; rng=default_rng())
  trace_prop = set(trace, rand(rng, prop(trace)))
  accept_prob = min(1, exp(logpdf(prog, trace_prop) - logpdf(prog, trace)))
  accept = rand(Bernoulli(accept_prob))
  new_trace = accept ? trace_prop : trace
  return accept, new_trace
end

propose_bias(trace) = TruncatedNormal(trace.bias, 0.05, 0, 1)
get_bias(trace) = trace.bias
set_bias(trace, bias) = (trace..., bias=bias)

function run_inference(trace, prog, num_steps)
  record = @NamedTuple{accepted::Bool, bias::Float64}[]
  for _ in 1:num_steps
    accepted, trace = mhsym(trace, prog, propose_bias, get_bias, set_bias)
    push!(record, (accepted=accepted, bias=get_bias(trace)))
  end
  return record, trace
end

# generate artificial data
# change true bias and number of observations as you like
n = 1000 # number of observations
true_bias = 0.5
data = rand(iid(biased_die(true_bias), n))
mean(iseven.(data))

# run inference
prog = multiple_rolls(n)
init_trace = (bias=0.5, numbers=data)
num_sample_steps = 100_000
@time record, trace = run_inference(init_trace, prog, num_sample_steps);

# inspect record
map(r -> r.accepted, record) |> mean
map(r -> r.bias, record) |> mean

# compute bayes_factor
using LogExpFunctions: logsumexp
@time data_logpdf_biased = let
  logpdfs = map(record) do r
    trace = (bias=r.bias, numbers=data)
    logpdf(prog, trace)
  end
  logsumexp(logpdfs) - log(num_sample_steps)
end

@probprog function multiple_unbiased_rolls(n)
  numbers ~ iid(DiscreteUniform(1, 6), n)
  return
end

data_logpdf_unbiased = logpdf(multiple_unbiased_rolls(n), (; numbers = data))
bayes_factor = exp(data_logpdf_biased - data_logpdf_unbiased)
log10(bayes_factor)

@probprog function biased_die_analytical(numodd, numeven)
  is_even ~ BetaBinomial(1, numeven, numodd)
  possible_numbers = is_even ? even_die_numbers : odd_die_numbers
  number ~ UniformCategorical(possible_numbers)
  return
end

fromtrace(::typeof(biased_die_analytical), trace) = trace.number
totrace(::typeof(biased_die_analytical), number) = 
  (is_even=iseven(number), number=number)

@probprog function multiple_rolls_analytical(numodd, numeven, n)
  numbers ~ iid(biased_die_analytical(numodd, numeven), n)
  return
end

numodd = count(isodd, data)
numeven = n - numodd
@time data_logpdf_biased_analytical = 
  logpdf(multiple_rolls_analytical(numodd, numeven, n), (; numbers=data))

bayes_factor_analytical = 
  exp(data_logpdf_biased_analytical - data_logpdf_unbiased)
log10(bayes_factor_analytical)