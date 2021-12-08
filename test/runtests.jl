using SimpleProbabilisticPrograms
using Test

using Distributions: Beta, Bernoulli

# first test
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

# test Dirichlet categorical
using LogExpFunctions; logsumexp
dc = DirCat(Dict("$i" => 2*i for i in 1:3))
@assert dc.pscounts |> values |> sum |> isapprox(12)
@assert !dc.logpdfs_uptodate
@assert log(0) < logpdf(dc, rand(dc)) < log(1)
@assert dc.logpdfs_uptodate
@assert dc.logpdf |> values |> logsumexp |> exp |> isapprox(1)
@assert dc.logvarpdf |> values |> logsumexp |> exp |> isapprox(1)
add_obs!(dc, "1", 1)
@assert dc.pscounts["1"] ≈ 3
@assert !dc.logpdfs_uptodate

# test observation with probprob
dc = (flat_dircat(collect("abc")), flat_dircat(1:4))
SimpleProbabilisticPrograms.update_logpdfs!(dc[1])
SimpleProbabilisticPrograms.update_logpdfs!(dc[2])
@probprog function observation_test(dc)
  char ~ dc[1]
  num  ~ dc[2]
  return
end
trace = rand(observation_test(dc))
@test dc[1].logpdfs_uptodate && dc[2].logpdfs_uptodate
add_obs!(observation_test(dc), trace, 1)
@test !dc[1].logpdfs_uptodate && !dc[2].logpdfs_uptodate
@test dc[1].pscounts[trace.char] == 2 && sum(values(dc[1].pscounts)) == 4
@test dc[2].pscounts[trace.num] == 2 && sum(values(dc[2].pscounts)) == 5