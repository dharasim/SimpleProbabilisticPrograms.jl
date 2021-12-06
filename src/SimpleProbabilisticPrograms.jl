# Values of this package:
# - zero cost abstraction for evaluation of logpdfs 
#   (achieved by type stability and using stack allocation where ever possible)
# - Simple an intuitive model specification 
#   (achieved by implementing Distributions.jl interface & using math notation)
# - Thin abstraction layer with composable inference methods
#   (so that it's easy to maintain and integrates well into the Julia ecosystem)
#
# Not prioritized:
# - implementation of out-of-the-box inference methods
#
# Info:
# - inspired by Pyro & Gen.jl
# - using Lens (optics like in Haskell) for achieving type stability in trace 
#   accesses and updates

module SimpleProbabilisticPrograms

using Random: AbstractRNG, default_rng
using MacroTools: @capture, splitdef, combinedef, postwalk, prewalk

import Base: show, rand
import Distributions: logpdf

export logpdf # re-export from Distributions.jl
export @probprog, fromtrace, totrace
export iid
export UniformCategorical

struct ProbProg{C, F, A, KW}
  construct::C
  run::F
  args::A
  kwargs::KW
end

show(io::IO, ::ProbProg) = print(io, "ProbProg(...)")

fromtrace(prog::ProbProg, trace) = fromtrace(prog.construct, trace)
totrace(prog::ProbProg, x) = totrace(prog.construct, x)

fromtrace(::Any, trace) = trace
totrace(::Any, x) = x

function logpdf(model::ProbProg, x)
  trace = totrace(model, x)
  interpreter, _ = interpret(model, EvalTrace(0.0, trace))
  interpreter.logpdf
end

function rand(rng::AbstractRNG, model::ProbProg) 
  fromtrace(model, interpret(model, RandTrace(rng)).interpreter.trace)
end

struct ProbProgSyntaxError <: Exception 
  msg :: String
end

"""
    @probprog function_definition

A macro that transforms a function definition into a smart constructor for
probabilistic programs. 

Probabilistic programs `prog` are distributions over their traced random 
samples and they implement a simple distribution interface.
Execute the program and record all random samples in a trace (a named tuple) 
with `rand(rng, prog)`. 
Evaluate the logarithm of a trace's probability with `logpdf(prog, trace)`.

# Example
```julia-repl
julia> using Distributions: Beta, Bernoulli

julia> @probprog function beta_bernoulli_model(a, b, n)
           bias ~ Beta(a, b)
           coins ~ iid(Bernoulli(bias), n)
           return
       end
model (generic function with 1 method)

julia> prog = beta_bernoulli_model(3, 4, 10)
ProbProg(...)

julia> x = rand(prog)
(bias = 0.254528229459866, coins = Bool[1, 1, 1, 0, 0, 1, 0, 1, 0, 0])

julia> logpdf(prog, x)
-7.781320601339095
```

# Syntax transformation
All sample statements, which must be declared using the syntax 
`name ~ distribution`, are transformed into calls of the `sample` function.
For example, `heads ~ Bernoulli(0.5)` is transformed into
`interpreter, heads = sample(interpreter, Bernoulli(0.5), get, set)` where
`get(trace) = trace.heads` and `set(trace, val) = (trace..., heads=val)`. 
The getter and setter are used by the interpreters of the program.
"""
macro probprog(ex)
  i = gensym() # generate unique symbol for the interpreter

  # Rewrite a single sample statement indicated by `~` into a call to the
  # sample method, which takes 4 arguments: an interpreter, 
  # a distribution (something that implements rand and logpdf), 
  # as well as a getter and a setter for the the sample site in the trace.
  function rewrite_sample_expr(ex)
    if @capture(ex, name_ ~ distribution_call_)
      get_ex = :(trace -> trace.$name)
      set_ex = :((trace, val) -> (trace..., $name = val))
      :(($i, $name) = SimpleProbabilisticPrograms.sample(
                        $i, $distribution_call, $get_ex, $set_ex))
    else
      ex
    end
  end

  # dictionary representing the function definition in expression `ex`
  def_dict = splitdef(ex)

  # add interpreter as first argument
  def_dict[:args] = [i, def_dict[:args]...]

  # rewrite all sample statements indicated by `~`
  def_dict[:body] = postwalk(rewrite_sample_expr, def_dict[:body])

  # test that last expression is not a sample expression
  if @capture(def_dict[:body].args[end], _ ~ _)
    throw(ProbProgSyntaxError("last expression cannot be a sample statement"))
  end

  # add interpreter to the original return expression
  def_dict[:body].args[end] = 
    if @capture(def_dict[:body].args[end], return r_ex_)
      :((interpreter=$i, return_val=$r_ex))
    else
      :((interpreter=$i, return_val=$(def_dict[:body].args[end])))
    end

  # The expression returned by the macro evaluates to a definition of a function
  # that constructs a `ProbProg` object. This mimics the construction of 
  # distributions in `Distributions.jl`.
  esc(
    quote
      function $(def_dict[:name])(args...; kwargs...) 
        run = let
          # wrapped in a let statement to avoid name conflict
          $(combinedef(def_dict))
        end
        SimpleProbabilisticPrograms.ProbProg(
          $(def_dict[:name]), run, args, kwargs)
      end
    end
  )
end

########################################
### iid distributed random variables ###
########################################

struct IID{D}
  dist::D
  n::Int
  
  function IID(dist::D, n) where D
    @assert n > 0
    new{D}(dist, n)
  end
end

"""
    iid(distribution, number_samples)

Generate multiple independent and identically distributed samples from 
a distribution.

# Example
```julia-repl
julia> using Distributions: Bernoulli

julia> rand(Bernoulli(0.4))
false

julia> rand(iid(Bernoulli(0.4), 5))
5-element Vector{Bool}:
 0
 1
 1
 0
 0
```
"""
iid(dist, n) = IID(dist, n)

logpdf(iid::IID, xs) = sum(logpdf(iid.dist, x) for x in xs)
rand(rng::AbstractRNG, iid::IID) = [rand(rng, iid.dist) for _ in 1:iid.n]

###########################
### Uniform Categorical ###
###########################

"""
    UniformCategorical(set_of_values)

A uniform distribution distribution over a set of values.

# Example
```julia-repl
julia> dist = UniformCategorical(Set(["uniform","categorical","distribution"]))
UniformCategorical{String}(Set(["distribution", "uniform", "categorical"]))

julia> x = rand(dist)
"uniform"

julia> exp(logpdf(dist, x))
0.3333333333333333
```
"""
struct UniformCategorical{T}
  values::Set{T}
end

function logpdf(dist::UniformCategorical, x) 
  x in dist.values ? log(1) - log(length(dist.values)) : log(0)
end

function rand(rng::AbstractRNG, dist::UniformCategorical)
  rand(rng, dist.values)
end

##############################################
### Interpreters of probabilistic programs ###
##############################################

"""
    Interpreter
    
Supertype for objects that can interpret probabilistic programs.

New interpreter types must implement 
`sample(interpreter, distribution, get_observation, set_observation)`.
Then, they can be used with `interpret(prog, interpreter)` automatically.
"""
abstract type Interpreter end

"""
    interpret(prog, interpreter) -> (interpreter', return_value)

Interpret a probabilistic program. This does not mutate the interpreter but
returns a (new) interpreter with possible changes alongside the return values
of the program.
"""
function interpret(prog::ProbProg, i=StdInterpreter()::Interpreter)
  prog.run(i, prog.args...; prog.kwargs...)
end

"""
    StdInterpreter()

Interpreter for probabilistic programs that performs random choices for sample
statements and nothing else.
"""
struct StdInterpreter <: Interpreter end

function sample(i::StdInterpreter, dist, get_obs, set_obs)
  x = rand(dist)
  return i, x
end

"""
    EvalTrace(trace)

Interpreter for probabilistic programs that computes the probability of a trace
by accumulating it sample statement by sample statement.
"""
struct EvalTrace{T} <: Interpreter
  logpdf::Float64
  trace::T
end

EvalTrace(trace) = EvalTrace(0.0, trace)

function addto_total_logpdf(i::EvalTrace, logpdf)
  EvalTrace(i.logpdf + logpdf, i.trace)
end

function sample(i::EvalTrace, dist, get_obs, set_obs)
  x = get_obs(i.trace)
  i = addto_total_logpdf(i, logpdf(dist, x))
  return i, x
end

"""
    RandTrace([rng])

Interpreter that runs a probabilistic program and traces the random choices
made in the sample statements in a named tuple.
"""
struct RandTrace{R,T} <: Interpreter
  rng::R
  trace::T
end

RandTrace(rng::AbstractRNG) = RandTrace(rng, (;))
RandTrace() = RandTrace(default_rng())

function sample(i::RandTrace, dist, get_obs, set_obs)
  x = rand(dist)
  new_trace = set_obs(i.trace, x)
  return RandTrace(i.rng, new_trace), x
end

end # module
