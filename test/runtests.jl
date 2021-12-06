using SimpleProbabilisticPrograms
using Test

using Distributions: Beta, Bernoulli

@probprog function beta_bernoulli_model(a, b, n)
  bias ~ Beta(a, b)
  coins ~ iid(Bernoulli(bias), n)
  return
end

model = beta_bernoulli_model(3, 4, 10)
trace = rand(model)
@test -Inf < logpdf(model, trace) < 0