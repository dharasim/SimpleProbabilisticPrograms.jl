using SimpleProbabilisticPrograms
using Test

using Distributions: Beta, Bernoulli, Dirac

@testset "basic tests" begin
  @probprog function beta_bernoulli_model(a, b, n)
    bias ~ Beta(a, b)
    coins ~ iid(Bernoulli(bias), n)
    return
  end
  model = beta_bernoulli_model(3, 4, 10)
  trace = rand(model)
  @test -Inf < logpdf(model, trace) < 0

  # test uniform categorical
  dist = UniformCategorical(Set(1:4))
  @test -Inf < logpdf(dist, rand(dist)) < 0

  # test Dirac distribution
  dist = Dirac(42)
  @test exp(logpdf(dist, rand(dist))) ≈ 1
end

# test overriding inverse bijections `fromtrace` and `totrace`
using Distributions: Categorical
@probprog function bijection_model(index_probs)
  index ~ Categorical(index_probs)
  return
end
index_probs = [0.5, 0.5]
model = bijection_model(index_probs)
@testset "trace bijection 1" begin
  @test rand(model) isa @NamedTuple{index::Int}
  @test log(0) < logpdf(model, rand(model)) < log(1)
end

import SimpleProbabilisticPrograms: fromtrace, totrace
const vals = ('a', 'b')
fromtrace(::typeof(bijection_model), trace) = vals[trace.index]
totrace(::typeof(bijection_model), val) = (;index=findfirst(isequal('a'), vals))
@testset "trace bijection 2" begin
  @test rand(model) isa Char
  @test log(0) < logpdf(model, rand(model)) < log(1)
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

@probprog function observation_test(dc)
  char ~ dc[1]
  num  ~ dc[2]
  return
end
@testset "add observations in probprogs" begin
  dc = (flat_dircat(collect("abc")), flat_dircat(1:4))
  SimpleProbabilisticPrograms.update_logpdfs!(dc[1])
  SimpleProbabilisticPrograms.update_logpdfs!(dc[2])
  trace = rand(observation_test(dc))
  @test dc[1].logpdfs_uptodate && dc[2].logpdfs_uptodate
  add_obs!(observation_test(dc), trace, 1)
  @test !dc[1].logpdfs_uptodate && !dc[2].logpdfs_uptodate
  @test dc[1].pscounts[trace.char] == 2 && sum(values(dc[1].pscounts)) == 4
  @test dc[2].pscounts[trace.num] == 2 && sum(values(dc[2].pscounts)) == 5
end

@probprog function rec_model(p)
  go_further ~ Bernoulli(p)
  if go_further
    next_level ~ rec_model(p)
  end
  return
end

@testset "recursive model" begin
  model = rec_model(0.3)
  for _ in 1:10
    @test log(0) < logpdf(model, rand(model)) < log(1)
  end
end