include("definitions.jl")

using Test

@testset "BasicTests" begin
    L = 4
    H = getHamiltonianAsMPO(L)
    ψ = randn(MatrixProductState{L, Complex{Float64}}, 100, 2)

    ϕ = compress(ψ, Dcut = 80) # compress Left

    # Contraction Test
    @test              ϕ'ϕ ≈ 1
    @test          ψ'ψ/ψ'ψ ≈ ϕ'ϕ
    @test ((ψ'*(H*ψ))/ψ'ψ) ≈ (ϕ' * (H * ϕ))/ϕ'ϕ
    @test ((ψ'*(H*ψ))/ψ'ψ) ≈ (ϕ' * (H * ψ))/ϕ'ψ

    # Simple Eigenvector test
    E₀, ϕ = searchGroundState(ψ, H)
    @test ϕ' * ((H * H) * ϕ) ≈ (ϕ' * H * ϕ)^2

    # Exact eigenvalue @ L=2
    L = 2
    H = getHamiltonianAsMPO(L)
    ψ = randn(MatrixProductState{L, Complex{Float64}}, 10, 2)
    ϕ = compress(ψ, Dcut = 8) # compress Left
    E₀, ϕ = searchGroundState(ψ, H)
    @test E₀ ≈ -3/4

    # Eigenvector-eigenvalue correspondence test
    Lexp  = ones(ComplexF64, 1, 1, 1)
    Renv = getRenv(ϕ, H)
    Hmat = getHamiltonianMatrix(Lexp, H[1], Renv[1])
    M = ϕ[1]
    @cast v[(α, i, j)] |= M[i, j, α]
    @test v' * Hmat * v ≈ E₀

    # Larger case with significant Dcut convergence test
    L = 20
    E₀ = 0
    H = getHamiltonianAsMPO(L)
    ψ = randn(MatrixProductState{L, Complex{Float64}}, 24, 2)
    ϕ = compress(ψ, Dcut = 8) # compress Left
    E₀, ϕ = searchGroundState(ψ, H)
    @test ϕ' * ((H * H) * ϕ) ≈ (ϕ' * (H * ϕ))^2
end

@testset "Infinite DMRG (2 sites)" begin
    model   = heisenbergModel
    L       = 100
    Dcut    = 16

    H, (E₀, φ), (Lenv, Renv), S = infiniteDMRG(model, Int(L/2); Dcut = Dcut)
    @test round(E₀) ≈ -44

    L = 12
    Dcut = 8
    H = getHamiltonianAsMPO(L)
    ψ = randn(MatrixProductState{L, Complex{Float64}}, 24, 2)
    ϕ = compress(ψ, Dcut = Dcut) # compress Left
    λ₀, ϕ = searchGroundState(ψ, H)

    H, (E₀, φ), (Lenv, Renv), S = infiniteDMRG(model, Int(L/2); Dcut = Dcut)

    @test ~(E₀ ≈ λ₀)

    Dcut = 16
    H, (E₀, φ), (Lenv, Renv), S = infiniteDMRG(model, Int(L/2); Dcut = Dcut)
    @test E₀ ≈ λ₀

    # @test φ' * ((H * H) * φ) ≈ (φ' * (H * φ))^2
end

@time let
    model   = heisenbergModel
    L       = 12
    Dcut    = 1000

    H, (E₀, φ), (Lenv, Renv), S = infiniteDMRG(model, Int(L/2); Dcut = Dcut)

    E₀
end

@time let
    L = 12
    E₀ = 0
    Dcut = 100
    H = getHamiltonianAsMPO(L)
    ψ = randn(MatrixProductState{L, Complex{Float64}}, Dcut, 2)
    ϕ = compress(ψ, Dcut = Dcut) # compress Left
    E₀, ϕ = searchGroundState(ψ, H)
end


@time let
    model   = heisenbergModel
    L       = 20
    Dcut    = 16
    maxiter = 10
    ϵ       = 1e-8
    Dschedule = [32, 36]

    E₀, ψ, Renv = DMRG(model, L; Dcut = Dcut, Dschedule = Dschedule, maxIteration = maxiter, ϵ = ϵ)

    H = getHamiltonianAsMPO(L)
    (E₀, ψ' * H * ψ)
end
