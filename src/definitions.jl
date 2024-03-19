using LinearAlgebra, LowRankApprox, KrylovKit
using TensorOperations
using TensorCast

struct Left end
struct Right end

Direction = Union{Left, Right}

function assertReal(x::Union{Complex, Real}; ϵ = 1e-10)
    if abs(imag(x)) > ϵ
        error("Error: x ∉ ℜ ↣ Im(x) = $(imag(x)) > $ϵ")
    end
    return real(x)
end

### --------------- MPS ---------------

# vector of rank-3 tensors | notation: ψ, ϕ, φ
struct MatrixProductState{L, T<:Number}
    tensors::Vector{Array{T,3}}
end

# ψ[i] ≡ ψ.tensors[i]
function Base.getindex(ψ::MatrixProductState, i::Int)
    return getindex(ψ.tensors, i)
end

function getMatrixProductState(v::Vector{Vector{T}}) where {T}
    L = length(v)
    tensors = [reshape(copy(v[i]), 1, 1, :) for i in 1:L]
    return MatrixProductState{L,T}(tensors)
end

function getMatrixProductState(v::Vector, L)
    return getMatrixProductState([v for _ in 1:L])
end

function compress(ψ::MatrixProductState{L, T}, direction::Direction; Dcut::Int = typemax(Int)) where {L, T}
    if direction === Right()
        return compressToRight(ψ, Dcut)
    else
        return compressToLeft(ψ, Dcut)
    end
end

function compress(ψ; Dcut)
    return compress(ψ, Left(); Dcut = Dcut)
end

function compressToRight(ψ::MatrixProductState{L, T}, Dcut::Int) where {L, T}
    tensors = Array{T, 3}[]

    # Eq. (31-33)
    B = ψ[1] # here ψ is a vector or rank-3 tensors (1st and last with dummy dimension)
    d = length(B[1, 1, :]) # physical dim
    # α - external (physical) indices, i,j - internal (matrix) indices
    @cast Bm[(α, i), j] |= B[i, j, α]   # reshape to matrix
    U, S, V = psvd(Bm, rank = Dcut)     # SVD
    @cast A[i, j, α] |= U[(α, i), j] (α ∈ 1:d) # reshape to rank-3 tensor
    push!(tensors, A) # collect A

    # Eq. (33-35)
    for i ∈ 2:L
        B = ψ[i] # get next tensor
        d = length(B[1, 1, :]) # physical dim
        # multiply S ̇V† with next tensor
        @tensor M[i, j, α] := (Diagonal(S) * V')[i, i′] * B[i′, j, α]
        @cast   Mm[(α, i), j] |= M[i, j, α] # reshape to matrix
        U, S, V = psvd(Mm, rank=Dcut)       # SVD
        @cast A[i, j, α] |= U[(α, i), j] (α ∈ 1:d) # reshape to rank-3 tensor
        push!(tensors, A) # collect A
    end

    return MatrixProductState{L, T}(tensors)
end

# Eq. (43¹/₂) [between 43 and 44]
function compressToLeft(ψ::MatrixProductState{L, T}, Dcut::Int) where {L, T}
    tensors = Array{T, 3}[]

    A = ψ[L]
    d = length(A[1, 1, :])

    @cast Am[i, (α, j)] |= A[i, j, α]
    U, S, V = psvd(Am, rank=Dcut)

    @cast B[i, j, α] |= V'[i, (α, j)] (α ∈ 1:d)
    push!(tensors, B)

    for i ∈ (L-1):-1:1
        A = ψ[i]
        d = length(A[1, 1, :])
        @tensor M[i, j, α]    := A[i, j′, α] * (U * Diagonal(S))[j′, j]
        @cast   Mm[i, (α, j)] |= M[i, j, α]

        U, S, V = psvd(Mm, rank=Dcut)

        @cast B[i, j, α] |= V'[i, (α, j)] (α ∈ 1:d)
        push!(tensors, B)
    end
    MatrixProductState{L, T}(reverse(tensors))
end

function compressToMixed(ψ::MatrixProductState{L, T}, l::Int, Dcut::Int) where {L, T}
    (l > 0 && l < L) || error("Error: 0 < l = $l < L = $L not satisfied!")

    tensorsA = Array{T, 3}[]

    B = ψ[1]
    d = length(B[1, 1, :])
    @cast Bm[(α, i), j] |= B[i, j, α]
    U, S, V = psvd(Bm, rank = Dcut)
    @cast A[i, j, α] |= U[(α, i), j] (α ∈ 1:d)
    push!(tensorsA, A)

    for i ∈ 2:l
        B = ψ[i]
        d = length(B[1, 1, :])
        @tensor M[i, j, α] := (Diagonal(S) * V')[i, i′] * B[i′, j, α]
        @cast   Mm[(α, i), j] |= M[i, j, α]
        U, S, V = psvd(Mm, rank=Dcut)
        @cast A[i, j, α] |= U[(α, i), j] (α ∈ 1:d)
        push!(tensorsA, A)
    end

    tensorsB = Array{T, 3}[]

    ⬥ = S

    A = ψ[L]
    d = length(A[1, 1, :])

    @cast Am[i, (α, j)] |= A[i, j, α]
    U, S, V = psvd(Am, rank=Dcut)

    @cast B[i, j, α] |= V'[i, (α, j)] (α ∈ 1:d)
    push!(tensorsB, B)

    for i ∈ (L-1):-1:(l+1)
        A = ψ[i]
        d = length(A[1, 1, :])
        @tensor M[i, j, α]    := A[i, j′, α] * (U * Diagonal(S))[j′, j]
        @cast   Mm[i, (α, j)] |= M[i, j, α]

        U, S, V = psvd(Mm, rank=Dcut)

        @cast B[i, j, α] |= V'[i, (α, j)] (α ∈ 1:d)
        push!(tensorsB, B)
    end

    return MatrixProductState{L, T}([tensorsA; tensorsB]), ⬥
end

# compress from Left to Right without Dcut
function toLeftCanonical(ψ)
    return compress(ψ, Right())
end

# compress from Right to Left without Dcut
function toRightCanonical(ψ)
    return compress(ψ, Left())
end

function getB(M::Array{T,3}) where {T}
    d = length(M[1, 1, :])
    @cast Mm[i, (α, j)] |= M[i, j, α]
    U, S, V = svd(Mm)
    @cast B[i, j, α] |= V'[i, (α, j)] (α ∈ 1:d)
    return B
end

# complex conjugate and swap matrix dimensions
function dagger(M::Array{T, 3}) where {T}
    return permutedims(conj.(M), (2, 1, 3))
end

# make random MPS with D×D matrix indices and d physical indices and
#   map to left-canonical then right-canonical
function Base.randn(::Type{MatrixProductState{L, T}}, D::Int, d::Int) where {L, T}
    tensors = [randn(1, D, d), [randn(D, D, d) for _ in 2:(L-1)]..., randn(D, 1, d)]
    return MatrixProductState{L, T}(tensors) |> toLeftCanonical |> toRightCanonical
end

function Base.length(::MatrixProductState{L, T}) where {L, T}
    return L
end

function Base.size(::MatrixProductState{L, T}) where {L, T}
    return (L,)
end

function Base.eltype(::Type{MatrixProductState{L, T}}) where {L, T}
    return T
end

function Base.copy(ψ::MatrixProductState{L, T}) where {L, T}
    return MatrixProductState{L,T}(copy.(ψ.tensors))
end

function Base.isequal(ψ::MatrixProductState, ϕ::MatrixProductState)
    return isequal(ψ.tensors, ϕ.tensors)
end

function Base.isapprox(ψ::MatrixProductState, ϕ::MatrixProductState)
    return isapprox(ψ.tensors, ϕ.tensors)
end

function Base.:(*)(ψ::MatrixProductState{L, T}, x::Number) where {L, T}
    return MatrixProductState{L,T}(ψ.tensors .* x)
end

function Base.:(*)(x::Number, ψ::MatrixProductState)
    return ψ * x
end

function Base.:(/)(ψ::MatrixProductState{L, T}, x::Number) where {L, T}
    return MatrixProductState{L, T}(ψ.tensors ./ x)
end

function Base.adjoint(ψ::MatrixProductState{L, T}) where {L,T}
    return Adjoint{T, MatrixProductState{L, T}}(ψ)
end

function Base.size(::Adjoint{T, MatrixProductState{L, T}}) where {L, T}
    return (1, L)
end

function Base.getindex(ψ::Adjoint{T, MatrixProductState{L, T}}, args...) where {L, T}
    out = getindex(reverse(ψ.parent.tensors), args...)
    return permutedims(conj.(out), (2, 1, 3))
end

function adjointTensors(ψ::MatrixProductState)
    return reverse(conj.(permutedims.(ψ.tensors, [(2, 1, 3)])))
end

### --------------- MPO ---------------

struct MatrixProductOperator{L, T<:Number}
    tensors::Vector{Array{T,4}}
end

function Base.getindex(O::MatrixProductOperator, args...)
    return getindex(O.tensors, args...)
end

function Base.length(::MatrixProductOperator{L, T}) where {L, T}
    return L
end

function dagger(M::Array{T, 4}) where {T}
    return permutedims(conj.(M), (2, 1, 3, 4))
end

### ---------- Contractions -----------

# Eq (95)
function Base.:(*)(ϕ⁺::Adjoint{T, MatrixProductState{L, T}}, ψ::MatrixProductState{L, T}) where {L, T}
    ϕ = ϕ⁺.parent

    M  = ψ.tensors[1]
    M̃⁺ = dagger(ϕ.tensors[1])
    @tensor contraction[j′, j] := M̃⁺[j′, 1, α] * M[1, j, α]

    for i in 2:L-1
        M  = ψ.tensors[i]
        M̃⁺ = dagger(ϕ.tensors[i])
        @tensor contraction[j′, j] := M̃⁺[j′, i′, α] * contraction[i′, i] * M[i, j, α]
    end

    M  = ψ.tensors[L]
    M̃⁺ = dagger(ϕ.tensors[L])
    result = @tensor M̃⁺[1, i′, α] * contraction[i′, i] * M[i, 1, α]

    return result
end

# Eq. (179)
function Base.:(*)(O::MatrixProductOperator{L, T}, ψ::MatrixProductState{L, T}) where {L, T}
    tensors = Array{T,3}[]
    for i in 1:L
        W = O.tensors[i]
        M = ψ.tensors[i]
        @reduce N[(i′, i), (j′, j), α] :=  sum(α′) W[i′, j′, α, α′] * M[i, j, α′]
        push!(tensors, N)
    end
    return MatrixProductState{L, T}(tensors)
end

# Conjugate Transpose of Eq. (179)
function Base.:(*)(ψ⁺::Adjoint{T,MatrixProductState{L,T}}, O::MatrixProductOperator{L, T}) where {L,T}
    ψ = ψ⁺.parent
    tensors = Array{T,3}[]
    W⁺ = dagger.(reverse(O.tensors))
    for i in 1:L
        W = W⁺[i]
        M = ψ.tensors[i]
        @reduce N[(i′, i), (j′, j), α] :=  sum(α′) W[i′, j′, α, α′] * M[i, j, α′]
        push!(tensors, N)
    end
    return adjoint(MatrixProductState{L, T}(tensors))
end

# Eq. (180)
function Base.:(*)(P::MatrixProductOperator{L, T}, O::MatrixProductOperator{L, T}) where {L, T}
    tensors = Array{T,4}[]
    for i in 1:L
        W̃ = P.tensors[i]
        W = O.tensors[i]
        @reduce V[(i′, i), (j′, j), α, α′] :=  sum(α″) W̃[i′, j′, α, α″] * W[i, j, α″, α′]
        push!(tensors, V)
    end
    return MatrixProductOperator{L, T}(tensors)
end

### ----------- Hamiltonian -----------

function getSpinOperators()
    O  = [0.0 0.0; 0.0 0.0]
    I  = [1.0 0.0; 0.0 1.0]
    S⁺ = [0.0 1.0; 0.0 0.0]
    S⁻ = [0.0 0.0; 1.0 0.0]
    Sᶻ = [0.5 0.0; 0.0 -0.5]
    return O, I, S⁺, S⁻, Sᶻ
end

# Eq. (184) Heisenberg Model (without ext. mag field)
function getHamiltonianGenerator(site::Int = 0)::Array{ComplexF64,4}
    J = 1.0;
    O, I, S⁺, S⁻, Sᶻ = getSpinOperators()
    result = zeros(ComplexF64, 5, 5, 2, 2)
    result[1, 1, :, :] = I;
    result[2, 1, :, :] = S⁻;
    result[3, 1, :, :] = S⁺;
    result[4, 1, :, :] = Sᶻ;
    result[5, 1, :, :] = O  *    J;
    result[5, 2, :, :] = S⁺ * 0.5J;
    result[5, 3, :, :] = S⁻ * 0.5J;
    result[5, 4, :, :] = Sᶻ *    J;
    result[5, 5, :, :] = I  *    J;
    return result
end

# Eq. (185-186)
function getHamiltonianAsMPO(L::Int = 2)::MatrixProductOperator{L, ComplexF64}
    L >= 2 || throw(DomainError(L, "$L < 2 sites"))

    W = [getHamiltonianGenerator() for _ in 1:L]
    W[1] = W[1][end:end, :, :, :]
    W[L] = W[L][:, 1:1, :, :]

    return MatrixProductOperator{L, ComplexF64}(W)
end

heisenbergModel = getHamiltonianAsMPO

### ---------- Measurements -----------

# calculate local operator (given as matrix / rank-2 tensor)

function getLocal(O::Array{T, 2}, ψ::MatrixProductOperator{L, T}, Lenv::Array{T, 3}, Renv::Array{T, 3}, S::Vector{T}) where {L, T}

    return nothing
end

### ----------- Environment -----------

# Eq. (197) [@tensoropt takes care of good multiplication order]
function iterateLexp(A, W, F) where {T}
    return @tensoropt F′[j″, j, j′] := F[i″, i, i′] * (conj.(A))[i, j, α] * W[i″, j″, α, β] * A[i′, j′, β]
end

# Eq. (197) in reverse
function iterateRexp(B, W, F) where {T}
    return @tensoropt F′[i″, i, i′] := (conj.(B))[i, j, α] * W[i″, j″, α, β] * B[i′, j′, β] * F[j″, j, j′]
end

# Eq. (197) iterate over all sites
function getRenv(ψ::MatrixProductState{L, T}, H::MatrixProductOperator{L, T}) where {L, T}
    Renv = Array{T, 3}[]
    Rexp = ones(T, 1, 1, 1)
    for l in L:-1:2
        Rexp = iterateRexp(ψ[l], H[l], Rexp)
        push!(Renv, Rexp)
    end
    return reverse(Renv)
end

function getEnvs(ψ::MatrixProductState{L, T}, H::MatrixProductOperator{L, T}, l::Int) where {L, T}
    (l >= 0 && l <= L) || error("Error: 0 <= l = $l <= L = $L not satisfied!")

    Lenv = Array{T, 3}[]
    Lexp = ones(T, 1, 1, 1)
    for i in 1:(l-1)
        Lexp = iterateLexp(ψ[i], H[i], Lexp)
        push!(Lenv, Lexp)
    end

    Renv = Array{T, 3}[]
    Rexp = ones(T, 1, 1, 1)
    for i in L:-1:(l+2)
        Rexp = iterateRexp(ψ[i], H[i], Rexp)
        push!(Renv, Rexp)
    end

    return Lenv, reverse(Renv)
end

### ----------- Eigen-Solve -----------

# Eq. (209¹/₂)
function getHamiltonianMatrix(Lexp::Array{T,3}, W::Array{T,4}, Rexp::Array{T,3}) where {T}
    @tensor H[α, i, j, α′, i′, j′] := Lexp[i″, i, i′] * W[i″, j″, α, α′] * Rexp[j″, j, j′]
    return @cast H[(α, i, j), (α′, i′, j′)] := H[α, i, j, α′, i′, j′]
end

# 2-site version of Eq. (209¹/₂)
function getHamiltonianMatrix(Lexp::Array{T,3}, W₁::Array{T,4}, W₂::Array{T,4}, Rexp::Array{T,3}) where {T}
    @tensor W₁₂[i″, j″, α, α′, β, β′] := W₁[i″, k, α, α′] * W₂[k, j″, β, β′]
    @tensor H[α, β, i, j, α′, β′, i′, j′] := Lexp[i″, i, i′] * W₁₂[i″, j″, α, α′, β, β′] * Rexp[j″, j, j′]
    return @cast H[(α, β, i, j), (α′, β′, i′, j′)] :=  H[α, β, i, j, α′, β′, i′, j′]
end

# Eq. (210)
function eigenSolve(direction::Direction, M::Array{T, 3}, Lexp::Array{T, 3}, W::Array{T, 4}, Rexp::Array{T, 3}) where {T}
    @cast v[(α, i, j)] |= M[i, j, α]
    H = getHamiltonianMatrix(Lexp, W, Rexp)
    E, V, _ = eigsolve(H, v, 1, :SR, ishermitian = true)
    λ₀ = E[1]
    v₀ = (V[1])
    if typeof(λ₀) != T
        λ₀ = T(λ₀)
        v₀ = Vector{T}(v₀)
    end
    return (λ₀, tensorSplit(direction, v₀, size(M))...)
end

# 2-site version of Eq. (210)
function eigenSolve(direction::Direction, M::Array{T, 4},
                    Lexp::Array{T, 3}, W₁::Array{T, 4}, W₂::Array{T, 4}, Rexp::Array{T, 3};
                    Dcut = 8, optimize = true) where {T}

    @cast v[(α, β, i, j)] |= M[i, j, α, β]

    H = getHamiltonianMatrix(Lexp, W₁, W₂, Rexp)
    E, V, I = if optimize
        # lanczos(Lexp, W₁, W₂, Rexp, M)
        eigsolve(H, v, 1, :SR; ishermitian = true, krylovdim = Dcut)
    else
        # lanczos(Lexp, W₁, W₂, Rexp, M)
        eigsolve(H, v, 1, :SR; ishermitian = true, tol = 1e-12, krylovdim = Dcut)
    end

    I.converged >= 1 || error("Not converged!")

    λ₀ = T(E[1])
    v₀ = Vector{T}(V[1])

    return (λ₀, tensorSplit(direction, v₀, size(M), Dcut)...)
end

function splitMixed(v₀::Vector, (Dᵢ, Dⱼ, dα, dβ); Dcut::Int = typemax(Int))
    @cast Mm[(α, i), (β, j)] := v₀[(α, β, i, j)] (i ∈ 1:Dᵢ, j ∈ 1:Dⱼ, α ∈ 1:dα, β ∈ 1:dβ)
    Mm = Array{ComplexF64, 2}(Mm)
    U, S, V = psvd(Mm, rank = Dcut)

    S /= sqrt(sum([s^2 for s in S]))

    @cast A[i, k, α] |= U[(α, i), k] (α ∈ 1:dα, i ∈ 1:Dᵢ)
    @cast B[k, j, β] |= V'[k, (β, j)] (β ∈ 1:dβ, j ∈ 1:Dⱼ)
    return A, S, B
end

function tensorSplit(direction::Direction, v₀, dims, Dcut::Int = 0)
    if direction === Right()
        if Dcut == 0
            return splitRight(v₀, dims)
        else
            return splitRight(v₀, dims, Dcut)
        end
    else
        if Dcut == 0
            return splitLeft(v₀, dims)
        else
            return splitLeft(v₀, dims, Dcut)
        end
    end
end

function splitRight(v₀::Vector, (Dᵢ, Dⱼ, d))
    @cast Mm[(α, i), j] := v₀[(α, i, j)] (i ∈ 1:Dᵢ, j ∈ 1:Dⱼ, α ∈ 1:d)
    U, S, V = svd(Mm)
    @cast A[i, j, α] |= U[(α, i), j] (α ∈ 1:d, i ∈ 1:Dᵢ, j ∈ 1:Dⱼ)
    return A, Diagonal(S) * V'
end

function splitLeft(v₀::Vector, (Dᵢ, Dⱼ, d))
    @cast Mm[i, (α, j)] |= v₀[(α, i, j)] (i ∈ 1:Dᵢ, j ∈ 1:Dⱼ, α ∈ 1:d)
    U, S, V = svd(Mm)
    @cast B[i, j, α] |= V'[i, (α, j)] (α ∈ 1:d)
    return U * Diagonal(S), B
end

function splitRight(v₀::Vector, (Dᵢ, Dⱼ, dα, dβ), Dcut)
    @cast Mm[(α, i), (β, j)] := v₀[(α, β, i, j)] (i ∈ 1:Dᵢ, j ∈ 1:Dⱼ, α ∈ 1:dα, β ∈ 1:dβ)
    Mm = Array{ComplexF64, 2}(Mm)
    U, S, V = psvd(Mm, rank = Dcut)

    S /= sqrt(sum([s^2 for s in S]))

    @cast A[i, k, α] |= U[(α, i), k] (α ∈ 1:dα)
    @cast M[k, j, β] |= (Diagonal(S) * V')[k, (β, j)] (β ∈ 1:dβ, j ∈ 1:Dⱼ)
    return A, M
end

function splitLeft(v₀::Vector, (Dᵢ, Dⱼ, dα, dβ), Dcut)
    @cast Mm[(α, i), (β, j)] := v₀[(α, β, i, j)] (i ∈ 1:Dᵢ, j ∈ 1:Dⱼ, α ∈ 1:dα, β ∈ 1:dβ)
    Mm = Array{ComplexF64, 2}(Mm)
    U, S, V = psvd(Mm, rank = Dcut)

    S /= sqrt(sum([s^2 for s in S]))

    @cast B[k, j, β] |= V'[k, (β, j)] (β ∈ 1:dβ, j ∈ 1:Dⱼ)
    @cast M[i, k, α] |= (U * Diagonal(S))[(α, i), k] (α ∈ 1:dα, i ∈ 1:Dᵢ)
    return M, B
end

### ----------- Lanczos MPS -----------

function Hv(Lexp::Array{T,3}, W₁::Array{T,4}, W₂::Array{T,4}, Rexp::Array{T,3}, v::Array{T,4}) where {T}
    # v[i, j, α, β]
    @tensor LH[k″, i, i′, α, α′] := Lexp[i″, i, i′] * W₁[i″, k″, α, α′]
    @tensor LHψψ[k″, j′, β′, i, α] := LH[k″, i, i′, α, α′] * v[i′, j′, α′, β′]
    @tensor LHψψH[j″, j′, i, α, β] := LHψψ[k″, j′, β′, i, α] * W₂[k″, j″, β, β′]

    # note: when applying the Hamiltonia to a state v in MPS form, what is the result?
    # is it v or v'? It look like it should be a conjugate. How to check that?

    @tensor Hv[i, j, α, β] := LHψψH[j″, j′, i, α, β] * Rexp[j″, j, j′]

    return Hv;
end

function lanczos(Lexp::Array{T,3}, W₁::Array{T,4}, W₂::Array{T,4}, Rexp::Array{T,3}, v₀::Array{T,4}; tol = 10^-12) where {T}
    Dᵢ, Dⱼ, dα, dβ = size(v₀)

    v₁ = Hv(Lexp, W₁, W₂, Rexp, v₀)

    @cast w₀[(α, β, i, j)] |= v₀[i, j, α, β]
    @cast w₁[(α, β, i, j)] |= v₁[i, j, α, β]

    a₀ = w₁'w₀
    w₁ -= a₀ * w₀

    b₁ = norm(w₁)
    b₁ > tol || return a₀, v₀

    w₁ /= b₁
    @cast v₁[i, j, α, β] |= w₁[(α, β, i, j)] (i ∈ 1:Dᵢ, j ∈ 1: Dⱼ, α ∈ 1:dα, β ∈ 1:dβ)

    v₂ = Hv(Lexp, W₁, W₂, Rexp, v₁)

    @cast w₂[(α, β, i, j)] |= v₂[i, j, α, β]

    a₁ = w₂'w₁

    M = assertReal.([a₀ b₁; b₁ a₁])
    f = eigen(M)

    E = [ f.values[1] ]
    V = [ conj(f.vectors[1,1]) * w₀ + conj(f.vectors[2,1]) * w₁ ]

    return E, V, true
end


### ---------- Sweep Updates ----------

# sweep to the right & then to left (=== total of 1¹/₂ sweep)
function warmup(ψ::MatrixProductState{L, T}, H::MatrixProductOperator{L, T},
                    Lenv::Vector{Array{T,3}}, Renv::Vector{Array{T,3}}, S::Vector{T};
                    Dcut = 8) where {L, T}
    E = zero(T)
    l = length(Lenv)

    l == length(Renv) || error("Error: left and right environments do not have equal sizes.")
    l == Int(L/2) || error("Error: Improper Lenv size l = $l. Relation 2*$l == $L = L not satisfied.")

    Lexp = Lenv[l]

    B = ψ[l+1]
    @tensor M[i, j, α] := Diagonal(S)[i, k] * B[k, j, α]

    for s in (l+1):(L-1)
        B = ψ[s+1]
        @tensor v[i, j, α, β] := M[i, k, α] * B[k, j, β]

        W₁, W₂ = H[s], H[s+1]

        if s < L - 1
            Rexp = Renv[L-1-s]
        else
            Rexp = ones(T, 1, 1, 1)
        end

        # diagonalize and SVD (with Dcut: rank -> Dcut)
        # note: don't converge fully (optimize -> false)
        E, A, M = eigenSolve(Right(), v, Lexp, W₁, W₂, Rexp;
                        Dcut = Dcut, optimize = false)

        ψ.tensors[s] = A

        Lexp = iterateLexp(A, W₁, Lexp)
        push!(Lenv, Lexp)
    end

    # Update last site
    ψ.tensors[L] = copy(M)

    Renv = Array{T, 3}[]
    Rexp  = ones(T, 1, 1, 1)


    for s in (L-1):-1:1
        A = ψ[s]
        @tensor v[i, j, α, β] := A[i, k, α] * M[k, j, β]

        W₁, W₂ = H[s], H[s+1]

        if s > 1
            Lexp = Lenv[s-1]
        else
            Lexp = ones(T, 1, 1, 1)
        end

        # diagonalize and SVD (with Dcut: rank -> Dcut)
        # note: don't converge fully (optimize -> false)
        E, M, B = eigenSolve(Left(), v, Lexp, W₁, W₂, Rexp;
                        Dcut = Dcut, optimize = false)

        ψ.tensors[s+1] = B

        Rexp = iterateRexp(B, W₂, Rexp)
        push!(Renv, Rexp)
    end
    # update first site (holds singular value info)
    ψ.tensors[1] = copy(M)

    # clean left environment
    Lenv = Array{T, 3}[]

    return E, ψ, Lenv, Renv
end


function sweep(direction::Direction, ψ::MatrixProductState{L, T}, H::MatrixProductOperator{L, T},
                Lenv::Vector{Array{T,3}}, Renv::Vector{Array{T,3}};
                Dcut = 8) where {L, T}

    # initialize GS energy
    E  = Vector{T}()
    E₀ = missing

    if direction === Right()
        # Lenv should be already empty from previous sweeps
        Lexp = ones(T, 1, 1, 1)

        M = ψ[1]
        for s in 1:(L-1)
            B = ψ[s+1]
            @tensor v[i, j, α, β] := M[i, k, α] * B[k, j, β]

            W₁, W₂ = H[s], H[s+1]

            if s < L - 1
                Rexp = Renv[L-1-s]
            else
                Rexp = ones(T, 1, 1, 1)
            end

            E₀, A, M = eigenSolve(Right(), v, Lexp, W₁, W₂, Rexp; Dcut = Dcut)

            push!(E, E₀)

            ψ.tensors[s] = A

            Lexp = iterateLexp(A, W₁, Lexp)
            push!(Lenv, Lexp)
        end
        # update last site (holds singular value info)
        ψ.tensors[L] = copy(M)

        # clean right environment
        Renv = Array{T, 3}[]
    else
        # Renv should be already empty from previous sweeps
        Rexp  = ones(T, 1, 1, 1)

        M = ψ[L]
        for s in (L-1):-1:1
            A = ψ[s]
            @tensor v[i, j, α, β] := A[i, k, α] * M[k, j, β]

            W₁, W₂ = H[s], H[s+1]

            if s > 1
                Lexp = Lenv[s-1]
            else
                Lexp = ones(T, 1, 1, 1)
            end

            E₀, M, B = eigenSolve(Left(), v, Lexp, W₁, W₂, Rexp; Dcut = Dcut)

            push!(E, E₀)

            ψ.tensors[s+1] = B

            Rexp = iterateRexp(B, W₂, Rexp)
            push!(Renv, Rexp)
        end
        # update first site (holds singular value info)
        ψ.tensors[1] = copy(M)

        # clean left environment
        Lenv = Array{T, 3}[]
    end

    return assertReal.(E), ψ, Lenv, Renv
end

function sweepRight!(ψ::MatrixProductState{L, T}, H::MatrixProductOperator{L, T}, Renv) where {L, T}
    Lenv = Array{T, 3}[]
    Lexp  = ones(T, 1, 1, 1)
    E = zero(T)
    for l in 1:(L-1)
        W = H[l]

        E, A, SV⁺ = eigenSolve(Right(), ψ[l], Lexp, W, Renv[l])
        ψ.tensors[l] = A

        Lexp = iterateLexp(A, W, Lexp)
        push!(Lenv, Lexp)

        B₁ = ψ.tensors[l+1]
        @tensor M₁[k, j, α] := SV⁺[k, i] * B₁[i, j, α]
        ψ.tensors[l+1] = M₁
    end
    return Lenv, E
end

function sweepLeft!(ψ::MatrixProductState{L, T}, H::MatrixProductOperator{L, T}, Lenv) where {L, T}
    Renv = Array{T, 3}[]
    Rexp  = ones(T, 1, 1, 1)
    E = zero(T)
    for l in L:-1:2
        W = H[l]

        E, US, B = eigenSolve(Left(), ψ[l], Lenv[l-1], W, Rexp)
        ψ.tensors[l] = B

        Rexp = iterateRexp(B, W, Rexp)
        push!(Renv, Rexp)

        A₁ = ψ.tensors[l-1]
        @tensor M₁[i, k, α] :=  A₁[i, j, α] * US[j, k]
        ψ.tensors[l-1] = M₁
    end
    return Renv, E
end


### ------- Ground State Search -------

function isEigen(ϕ::MatrixProductState, H::MatrixProductOperator; ϵ = 1e-8)
    ϕ = toRightCanonical(ϕ)
    return isapprox(ϕ' * (H * (H * ϕ)), (ϕ' * (H * ϕ))^2, rtol = ϵ)
end

function searchGroundState(ψ::MatrixProductState{L, T}, H::MatrixProductOperator{L, T}; maxIteration = 10, ϵ = 1e-8) where {L, T}
    iteration = 0
    E₀        = zero(T)
    ϕ         = copy(ψ)

    println("Ground State Search")
    println("> Computing R expressions")
    Renv = getRenv(ψ, H)

    enable_cache(maxsize = 8*10^9)
    while true
        iteration += 1

        println(">> Performing right sweep: $iteration")
        Lenv, E₀′ = sweepRight!(ϕ, H, Renv)

        println(">> Performing left sweep: $iteration")
        Renv, E₀  = sweepLeft!(ϕ, H, Lenv)

        println(E₀′, ", ", E₀)

        if iteration >= maxIteration
            @warn "Did not converge in $maxIteration iterations!"
            break
        elseif isEigen(ϕ, H, ϵ = ϵ)
            println("Converged in $iteration iterations!")
            break
        elseif iteration > 1 && E₀ ≈ E₀′
            @warn "Stuck in local minimum after $iteration iterations!"
            break
        end
    end
    println("Ground State Search END\n")
    clear_cache()

    return assertReal(E₀), ϕ
end


### ------- Infinite-DMRG 2-site ------

function infiniteDMRG(model::Function, nSteps::Int; Dcut::Int = 8)
    (nSteps > 0) || error("Error: number of steps must be a positive integer!")
    (Dcut > 0)   || error("Error: keyword parameter Dcut must be a positive integer!")

    maxL        = 2 * nSteps
    T           = ComplexF64

    # initialize model as MPO (empty)
    H = MatrixProductOperator{maxL, T}([])

    # initialize GS
    ψ = MatrixProductState{maxL, T}([])

    # initialize left and right environment (empty)
    Lenv = Array{T, 3}[]
    Renv = Array{T, 3}[]
    Lexp = ones(T, 1, 1, 1)
    Rexp = ones(T, 1, 1, 1)

    # initialize matrix dimensions
    Dᵢ = 1
    Dⱼ = 1

    # initialize GS
    λ₀ = missing
    v₀ = missing

    # placeholder for final singular values
    S = Vector{T}[]

    # iterate over pairs of sites to add
    for l in 1:nSteps
        # add 2 sites
        H₁ = getHamiltonianGenerator(l)
        H₂ = getHamiltonianGenerator(maxL+1-l)
        if l == 1
            H₁ = H₁[end:end, :, :, :]
            H₂ = H₂[:, 1:1, :, :]
        end
        insert!(H.tensors, l,   H₁)
        insert!(H.tensors, l+1, H₂)

        # obtain physical dimensions at new sites
        dα = size(H₁)[3]
        dβ = size(H₂)[3]

        # solve model exactly
        Hm = getHamiltonianMatrix(Lexp, H₁, H₂, Rexp)
        E, V, _ =
        eigsolve(
            Hm,                 # Hamiltonian matrix
            Dᵢ * Dⱼ * dα * dβ,  # length of initial vector
            1,                  # min number of eigenvalues to converge
            :SR,                # obtain smallest real component first
            ishermitian = true  # runs Lanczos (instead of Arnoldi)
        )

        # obtain GS (& set eltype = T)
        λ₀ = T(E[1])
        v₀ = Vector{T}(V[1])

        # SVD on GS and reshape
        A, S, B = splitMixed(v₀, (Dᵢ, Dⱼ, dα, dβ); Dcut = Dcut)

        # update GS
        insert!(ψ.tensors, l,   A)
        insert!(ψ.tensors, l+1, B)

        # absorb A (B) to Left (Right) environment
        Lexp = iterateLexp(A, H₁, Lexp)
        Rexp = iterateRexp(B, H₂, Rexp)
        push!(Lenv, Lexp)
        push!(Renv, Rexp)

        (Dᵢ, Dⱼ) = (size(A)[2], size(B)[1])
    end

    return H, (assertReal(λ₀), ψ), (Lenv, Renv), Vector{T}(S)
end


function finiteDMRG(ψ::MatrixProductState{L, T}, H::MatrixProductOperator{L, T},
                    Lenv::Vector{Array{T, 3}}, Renv::Vector{Array{T, 3}}, S::Vector{T};
                    Dcut = 8, Dschedule = [8], maxIteration = 10, ϵ = 1e-8, verbose = false) where {L, T}

    ~verbose || @info "Iterative Search"

    ~verbose || @info "Performing warmup sweep"
    λ₀, ϕ, Lenv, Renv = warmup(ψ, H, Lenv, Renv, S; Dcut = Dcut)

    iteration = 0
    E         = missing
    Elist     = []

    # actuall sweeping
    while true
        iteration += 1

        if length(Dschedule) >= iteration
            Dcut = Dschedule[iteration]
        end

        ~verbose || @info "Performing right sweep: $iteration"
        E′, ϕ, Lenv, Renv = sweep(Right(), ϕ, H, Lenv, Renv; Dcut = Dcut)

        ~verbose || @info "Performing left sweep: $iteration"
        E, ϕ, Lenv, Renv  = sweep(Left(), ϕ, H, Lenv, Renv; Dcut = Dcut)

        @time let
            l = Int(L/2)

            Rexp = Renv[L-1]
            Rexp = iterateRexp(ϕ[1], H[1], Rexp)

            Lexp = ones(T, 1, 1, 1)

            @tensoropt EGS = Lexp[i″, i, i′] * Rexp[i″, i, i′]

            println(EGS |> assertReal)
        end

        push!(Elist, E′)
        push!(Elist, E)

        if iteration >= maxIteration
            ~verbose || @warn "Did not converge in $maxIteration iterations!"
            break
        elseif isEigen(ψ, H, ϵ = ϵ)
            ~verbose || @info "Converged in $iteration iterations!"
            break
        # elseif iteration > 1 && E₀ ≈ E₀′
        #     ~verbose || @warn "Stuck in local minimum after $iteration iterations!"
        #     break
        end
    end
    ~verbose || @info "Iterative Search END\n"

    return Elist, ψ, Renv
end


# DMRG
function DMRG(model::Function, L::Int;
                Dcut = 8, Dschedule = [8], maxIteration = 10, ϵ = 1e-8, verbose = true)

    (L > 0 && mod(L, 2) == 0) ||
    error("Error: provided system size L = $L is not a positive even number!")

    # infinite DMRG
    H, (λ₀, ψ), (Lenv, Renv), S = infiniteDMRG(model, Int(L/2); Dcut = Dcut)

    # finite DMRG
    E, ψ, Renv = finiteDMRG(
        ψ, H, Lenv, Renv, S;
        Dcut = Dcut, Dschedule = Dschedule,
        maxIteration = maxIteration, ϵ = ϵ,
        verbose = verbose
    )

    return E, ψ, Renv
end

nothing
