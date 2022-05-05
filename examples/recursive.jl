using SimpleProbabilisticPrograms
using Distributions: Bernoulli

struct Flagged{T}
  isterminal :: Bool
  value      :: T
end

T(value) = Flagged(true, value)
NT(value) = Flagged(false, value)
T(x::Flagged) = Flagged(true, x.value)
NT(x::Flagged) = Flagged(false, x.value)
isterminal(x::Flagged) = x.isterminal

@probprog function rewrite(params, nt)
  terminate ~ Bernoulli(params.termination_prob)
  if terminate
    out ~ Dirac(T(nt))
  else
    ratio ~ DirCat(params.ratio_pscounts)
    out ~ Dirac((NT(nt.value*ratio), NT(1 - nt.value*ratio)))
  end
  return
end

params = (termination_prob=0.5, ratio_pscounts=Dict(1//2=>1, 1//3=>1))

rand(rewrite(params, NT(1)))

@probprog function unfold(termination_kernel, progression_kernel, nt)
  terminate ~ rand(termination_kernel(nt))
  if terminate
    label ~ Dirac(T(nt))
  else
    
  end
end