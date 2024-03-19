include("definitions.jl")

using DelimitedFiles

model   = heisenbergModel
L       = 16
Dcut    = 8
maxiter = 10
ϵ       = 1e-6
Dschedule = [12, 16, 20]

@time E, ψ, Renv = DMRG(model, L; Dcut = Dcut, Dschedule = Dschedule, maxIteration = maxiter, ϵ = ϵ)

writedlm("./data/conv.txt", E)
