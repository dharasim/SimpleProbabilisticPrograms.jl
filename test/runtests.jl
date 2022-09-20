using SimpleProbabilisticPrograms
using Test

using Random: MersenneTwister
using Distributions: Beta, Bernoulli
using Statistics: mean

@testset "basic tests" begin
  @probprog function beta_bernoulli_model(a, b, n)
    bias ~ Beta(a, b)
    coins ~ iid(Bernoulli(bias), n)
    return (; bias, coins)
  end
  model = beta_bernoulli_model(3, 4, 10)
  trace = rand(MersenneTwister(42), model)
  trace_static = rand(MersenneTwister(42), beta_bernoulli_model(3, 4, Val(10)))
  @test trace_static == trace
  @test -Inf < logpdf(model, trace) < 0
  @test insupport(model, trace)

  # test uniform categorical
  dist = UniformCategorical(Set(1:4))
  @test -Inf < logpdf(dist, rand(dist)) < 0
  @test insupport(dist, rand(dist))

  # test Dirac distribution
  dist = Dirac(42)
  @test exp(logpdf(dist, rand(dist))) ≈ 1
end

# test overriding inverse bijections `fromtrace` and `totrace`
using Distributions: Categorical
@probprog function bijection_model(index_probs)
  index ~ Categorical(index_probs)
  return (; index)
end
model = bijection_model([0.5, 0.5])
@testset "trace bijection 1" begin
  @test rand(model) isa @NamedTuple{index::Int}
  @test log(0) < logpdf(model, rand(model)) < log(1)
  @test insupport(model, rand(model))
end

import SimpleProbabilisticPrograms: recover_trace
const vals = ('a', 'b')
@probprog function bijection_model(index_probs)
  index ~ Categorical(index_probs)
  return vals[index]
end
recover_trace(::ProbProg{:bijection_model}, x) = (; index=findfirst(isequal(x), vals))
@testset "trace bijection 2" begin
  model = bijection_model([0.6, 0.4])
  @test rand(model) isa Char
  @test rand(model) in vals
  @test log(0) < logpdf(model, rand(model)) < log(1)
  @test insupport(model, rand(model))
end

using LogExpFunctions; logsumexp
@testset "Dirichlet categorical" begin
  dc = DirCat(Dict("$i" => 2*i for i in 1:3))
  @test dc.pscounts |> values |> sum |> isapprox(12)
  @test !dc.logpdfs_uptodate
  @test log(0) < logpdf(dc, rand(dc)) < log(1)
  @test dc.logpdfs_uptodate
  @test dc.logpdf |> values |> logsumexp |> exp |> isapprox(1)
  @test dc.logvarpdf |> values |> logsumexp |> exp |> isapprox(1)
  add_obs!(dc, "1", 1)
  @test dc.pscounts["1"] ≈ 3
  @test !dc.logpdfs_uptodate
end

@testset "add observations in probprogs" begin
  @probprog function observation_test(dc)
    char ~ dc[1]
    num  ~ dc[2]
    return (; char, num)
  end
  dc = (symdircat(collect("abc")), symdircat(1:4))
  SimpleProbabilisticPrograms.update_logpdfs!(dc[1])
  SimpleProbabilisticPrograms.update_logpdfs!(dc[2])
  trace = rand(observation_test(dc))
  @test dc[1].logpdfs_uptodate && dc[2].logpdfs_uptodate
  add_obs!(observation_test(dc), trace, 1)
  @test !dc[1].logpdfs_uptodate && !dc[2].logpdfs_uptodate
  @test dc[1].pscounts[trace.char] == 2 && sum(values(dc[1].pscounts)) == 4
  @test dc[2].pscounts[trace.num] == 2 && sum(values(dc[2].pscounts)) == 5
end

@testset "recursive model" begin
  @probprog function rec_model(p)
    go_further ~ Bernoulli(p)
    if go_further
      next_level ~ rec_model(p)
      return (; go_further, next_level)
    else
      return (; go_further)
    end
  end

  model = rec_model(0.3)
  for _ in 1:10
    @test log(0) < logpdf(model, rand(model)) < log(1)
    @test insupport(model, rand(model))
  end
end

@testset "simple conditional" begin
  cond = DictCond('a' => symdircat(1:3), 'b' => symdircat(10:15))
  @test rand(cond('a')) in 1:3
  @test rand(cond('b')) in 10:15
end

@testset "Monte Carlo" begin
  @probprog function bbm(a, b, n) # beta bernoulli model
    bias ~ Beta(a, b)
    coins ~ iid(Bernoulli(bias), n)
    return (; bias, coins)
  end

  N = 1000
  model = bbm(1, 1, N)
  data = rand(iid(Bernoulli(0.4), N))
  E = montecarlo(condition(model, on=(; coins=data)), LikelihoodWeighting(10_000)) 
  @test E(trace -> trace.bias) ≈ mean(data) atol=0.01

  # TODO: Test montecarlo with multivariate function
end


# t1 = (a=1, b=(c=2, d=3, e=4))
# t2 = (b=(c=5, d=6), f=7)
# @test mergetraces(t1, t2) == (a=1, b=(c=5, d=6, e=4), f=7)

# T1 = typeof(t1)
# T2 = typeof(t2)
# fieldnames(T1)
# fieldtypes(T1)
# fieldnames(T2)
# fieldtypes(T2)

# all_names = union(fieldnames(T1), fieldnames(T2))
# map(all_names) do n
#   if n in fieldnames(T1)
#     if n in fieldnames(T2)
#       if 
#     else
#       :($n = t1.$n)
#     end
#   else
#     :($n = t2.$n)
#   end
# end