import MLJBase
using LinearAlgebra: norm, I, pinv, eigen, tr, Diagonal, mul!, diagind, svd
using Flux: onehotbatch, onecold
using MLJ: levels, nrows
using CategoricalArrays: CategoricalArray
using CUDA
using TimerOutputs
using Infiltrator

mutable struct MIMSSVMClassifier <: MLJBase.Deterministic
    M_cut::Array{UnitRange{Int64},1} # How to "cut" X into multimodal data
    δ::Float64
    τ_1::Float64 ## impact of group 1 norm.
    τ_2::Float64 ## impact of trace norm.
    C::Float64
    μ::Float64
    ρ::Float64
    maxiter::Int64
    tol::Float64
    use_CUDA::Bool
    exact::Bool
end

mutable struct mimmsvm_vars
    # Original vars
    X::Array{Float64, 2}
    X_cut::Array{UnitRange{Int64}, 1}
    Y::Array{Float64, 2}
    W::Array{Float64, 2}
    b::Array{Float64, 1}

    # Introduced vars
    E::Array{Float64, 2}
    Q::Array{Float64, 2}
    R::Array{Float64, 2}
    T::Array{Float64, 2}
    U::Array{Float64, 2}
    V::Array{Float64, 2}
    D_1s::Vector{Array{Float64, 2}}
    D_2::Array{Float64, 2}

    # Lagrangian Multipliers
    Λ::Array{Float64, 2}
    Σ::Array{Float64, 2}
    Ω::Array{Float64, 2}
    Θ::Array{Float64, 2}
    Ξ::Array{Float64, 2}
    Γ::Array{Float64, 2}

    # Auxilary vars
    μ::Float64
    YI::Array{Float64, 2}
    WmX::Array{Float64, 2}
    WyX::Array{Float64, 2}
    by::Array{Float64, 2}

    # Note: the residual arrays can also be used as temporary storage
    E_res::Array{Float64, 2}
    Q_res::Array{Float64, 2}
    R_res::Array{Float64, 2}
    T_res::Array{Float64, 2}
    U_res::Array{Float64, 2}
    V_res::Array{Float64, 2}

    # To speed up W update (https://math.stackexchange.com/questions/670649/efficient-diagonal-update-of-matrix-inverse)
    Ps::Array{Array{Float64, 2}}
    Ds::Array{Diagonal}
    P⁻¹s::Array{Array{Float64, 2}}

    # Temporary arrays
    tmp1d::Array{Float64, 2} # A temporary 1 × d Array
    tmpdd::Array{Float64, 2} # A temporary d × d Array
    tmpdd2::Array{Float64, 2} # A second temporary d × d Array
    tmpKd::Array{Float64, 2} # A temporary K × d Array
    tmpKd2::Array{Float64, 2} # A second temporary K × d Array
    tmpKNI::Array{Float64, 2} # A temporary K × NI Array
    tmpKprime::Array{Array{Float64, 2}} # K temporary K × NIprime Arrays

    # vars for inexact W update
    s::Array{Float64, 1}
    ∇W::Array{Float64, 2}
    prime_sum::Array{Int64, 1}
    prime_cut::Array{UnitRange{Int64}, 1}
    Xᵀprime::Array{Array{Float64, 2}}
    XXᵀ_KXXᵀprime::Array{Array{Float64, 2}}

    # vars for dim
    K::Int
    G::Int
end

function mimmsvm_vars(; X, X_cut, Y, W, b, E, Q, R, T, U, V, D_1s, D_2, Λ, Σ, Ω, Θ, Ξ, Γ, μ, YI, WmX, WyX, by, E_res, Q_res, R_res, T_res, U_res, V_res, Ps, Ds, P⁻¹s, tmp1d, tmpdd, tmpdd2, tmpKd, tmpKd2, tmpKNI, tmpKprime, s, ∇W, prime_sum, prime_cut, Xᵀprime, XXᵀ_KXXᵀprime, K, G)
    v = mimmsvm_vars(X, X_cut, Y, W, b, E, Q, R, T, U, V, D_1s, D_2, Λ, Σ, Ω, Θ, Ξ, Γ, μ, YI, WmX, WyX, by, E_res, Q_res, R_res, T_res, U_res, V_res, Ps, Ds, P⁻¹s, tmp1d, tmpdd, tmpdd2, tmpKd, tmpKd2, tmpKNI, tmpKprime, s, ∇W, prime_sum, prime_cut, Xᵀprime, XXᵀ_KXXᵀprime, K, G)
end

function init_vars(model::MIMSSVMClassifier, _X, _y)
    #println("Starting init vars")
    N = length(_X) ## 416
    K = length(levels(_y)) ## 1
    G = length(model.M_cut)

    #println("sorting bags...")
    # Sort bags by class to ensure regular memory access pattern during prime indexing (see prime_cut)
    perm = sortperm(_y) 
    sort_y = _y[perm] 
    sort_X = _X[perm] 

    #println("building X...")
    X = hcat([MLJBase.matrix(x)' for x in sort_X]...) 
    @assert !any(isnan, X)
    nis = [size(x, 1) for x in sort_X] 
    X_cut = [sum(nis[1:ni])-nis[ni]+1:sum(nis[1:ni]) for ni in 1:N] 
    #println("building y cut...")
    Y = onehotbatch(sort_y, levels(sort_y)) .* 2.0 .- 1.0 

    d, NI = size(X) 
    W = randn(d, K) 
    b = randn(K) 

    #println("building introduced vars...")
    # Introduced vars
    E = randn(K, N)
    Q = randn(K, N)
    R = randn(K, N)
    T = randn(K, NI)
    U = randn(K, NI) 
    V = randn(d, K) 
    D_1s = [Matrix(1.0 * I, K, K) for g in 1:G] ## ref: https://stackoverflow.com/questions/57270276/identity-matrix-in-julia
    D_2 = Matrix(1.0 * I, d, d)
    #println("building lagrangian vars...")
    # Lagrangian Multipliers
    Λ = zeros(K, N)
    Σ = zeros(K, N)
    Ω = zeros(K, N)
    Θ = zeros(K, NI)
    Ξ = zeros(K, NI)
    Γ = zeros(d, K)

    #println("calculating auxilary vars...")
    # Auxilary vars
    μ = model.μ
    YI = hcat([repeat(Y[:,i], outer=(1, length(cut))) for (i, cut) in zip(1:N, X_cut)]...) ## 1×4160 Matrix{Float64}: 1.0  1.0  1.0  1.0 ...
    WmX = randn(K, NI)
    WyX = randn(K, NI)
    by = randn(K, NI)
    E_res = zeros(size(E))
    Q_res = zeros(size(Q))
    R_res = zeros(size(R))
    T_res = zeros(size(T))
    U_res = zeros(size(U))
    V_res = zeros(size(V))

    Ps = [randn(1, 1)] 
    Ds = [Diagonal(randn(1))] 
    P⁻¹s = [randn(1, 1)]

    #println("calculating auxilary vars for inexact update...")
    # Inexact vars
    s = randn(K) 
    ∇W = randn(size(W)) 
    XXᵀ = X * X' 
    prime_sum = vec(sum(YI .> 0, dims=2)) 
    prime_cut = [sum(prime_sum[1:m])-sum(prime_sum[m])+1:sum(prime_sum[1:m]) for m in 1:K]
    Xᵀprime = [X[:,cut]' for cut in prime_cut] 
    XXᵀ_KXXᵀprime = [XXᵀ .+ K .* X[:,cut]*X[:,cut]' for cut in prime_cut] 

    # Temporary arrays
    tmp1d = zeros(1, d)
    tmpdd = zeros(d, d)
    tmpdd2 = zeros(d, d)
    tmpKd = zeros(K, d)
    tmpKd2 = zeros(K, d)
    tmpKNI = zeros(K, NI)
    tmpKprime = [zeros(K, size(p, 2)) for p in Xᵀprime] 

    if CUDA.functional() && model.use_CUDA
        CUDA.allowscalar(false)
        X = convert(Array{Float64, 2}, X)

        v = mimmsvm_vars(CuArray(X), CuArray(X_cut), CuArray(Y), CuArray(W), CuArray(b), 
                       CuArray(E), CuArray(Q), CuArray(R), CuArray(T), CuArray(U), CuArray(V), D_1s,
                       CuArray(D_2), CuArray(Λ), CuArray(Σ), CuArray(Ω), CuArray(Θ), CuArray(Ξ), CuArray(Γ), μ, 
                       CuArray(YI), CuArray(WmX), CuArray(WyX), CuArray(by), CuArray(E_res),
                       CuArray(Q_res), CuArray(R_res), CuArray(T_res), CuArray(U_res), CuArray(V_res),
                       [CuArray(P) for P in Ps], Ds, [CuArray(P) for P in P⁻¹s], CuArray(tmp1d), CuArray(tmpdd), CuArray(tmpdd2),
                       CuArray(tmpKd), CuArray(tmpKd2), CuArray(tmpKNI), [CuArray(p) for p in tmpKprime], CuArray(s), CuArray(∇W), 
                       CuArray(prime_sum), CuArray(prime_cut), [CuArray(x) for x in Xᵀprime], [CuArray(x) for x in XXᵀ_KXXᵀprime], K, G)
    else
        v = mimmsvm_vars(X=X, X_cut=X_cut, Y=Y, W=W, b=b, E=E, Q=Q, R=R, T=T, U=U, V=V, D_1s=D_1s, D_2=D_2, Λ=Λ, Σ=Σ, Ω=Ω, Θ=Θ, Ξ=Ξ, Γ=Γ, μ=μ, YI=YI, WmX=WmX, WyX=WyX, by=by, E_res=E_res, Q_res=Q_res, R_res=R_res, T_res=T_res, U_res=U_res, V_res=V_res, Ps=Ps, Ds=Ds, P⁻¹s=P⁻¹s, tmp1d=tmp1d, tmpdd=tmpdd, tmpdd2=tmpdd2, tmpKd=tmpKd, tmpKd2=tmpKd2, tmpKNI=tmpKNI, tmpKprime=tmpKprime, s=s, ∇W=∇W, prime_sum=prime_sum, prime_cut=prime_cut, Xᵀprime=Xᵀprime, XXᵀ_KXXᵀprime=XXᵀ_KXXᵀprime, K=K, G=G)
    end

    # if model.exact
    #     calc_PDPs!(v)
    # end

    calc_WmX_WyX!(v)
    calc_by!(v)

    return v
end

# function g1_norm(model::MIMSSVMClassifier, v::mimmsvm_vars)
#     g1_norm = 0.0
#     for m in 1:length(model.M_cut)
#         cut = model.M_cut[m]

#         for col in 1:v.K
#             g1_norm += norm(v.V[cut, col], 2)
#         end
#     end

#     return g1_norm
# end

# function trace_norm(model::MIMSSVMClassifier, v::mimmsvm_vars)
#     return tr(sqrt.(v.W' * v.W))
# end

function p_exponent_matrix(matrix::Matrix, p::Float64 = -0.5)
    F = svd(matrix; full= false)
    # @infiltrate cond = any(isnan, F.U) || any(isnan, F.S) || any(isnan, F.Vt)
    return F.U * Diagonal(F.S.^p) * F.Vt
end

function check_nan(v::mimmsvm_vars; model = 1)
    @infiltrate cond = any(isnan, v.W)
    @infiltrate cond = any(isnan, v.b)
    @infiltrate cond = any(isnan, v.E)
    @infiltrate cond = any(isnan, v.Q)
    @infiltrate cond = any(isnan, v.R)
    @infiltrate cond = any(isnan, v.T)
    @infiltrate cond = any(isnan, v.U)
    @infiltrate cond = any(isnan, v.V)
    for g in 1:v.G
        @infiltrate cond = any(isnan, v.D_1s[g])
    end
    @infiltrate cond = any(isnan, v.Λ)
    @infiltrate cond = any(isnan, v.Σ)
    @infiltrate cond = any(isnan, v.Ω)
    @infiltrate cond = any(isnan, v.Θ)
    @infiltrate cond = any(isnan, v.Ξ)
    @infiltrate cond = any(isnan, v.Γ)
    return 0
end

function obj_loss(model::MIMSSVMClassifier, v::mimmsvm_vars)
    𝓛 = 0.0
    𝓛 += 0.5 * norm(v.W, 2) ^ 2.0
    𝓛 += model.C * sum(max.(1 .- (bag_max!(v.Q_res, v.WmX .+ v.b, v.X_cut) .- bag_max!(v.R_res, v.WyX .+ v.by, v.X_cut)).*v.Y, 0))

    for m in 1:length(model.M_cut)
        cut = model.M_cut[m]
        for col in 1:v.K
            𝓛 += model.τ_1 * norm(v.V[cut, col], 2)
        end
    end
    𝓛 += model.τ_2 * tr(p_exponent_matrix(v.W' * v.W, 0.5))

    return 𝓛
end

function lagrangian_loss(model::MIMSSVMClassifier, v::mimmsvm_vars; target = "all")
    𝓛 = 0.0
    if target == "all"
        𝓛 += 0.5 * norm(v.W, 2)^2.0
        𝓛 += model.C * sum(max.(v.Y .* v.E, 0))

        for g in 1:length(model.M_cut)
            V_g = v.V[model.M_cut[g], :]
            𝓛 += model.τ_1 * tr(V_g * v.D_1s[g] * V_g')
        end
        𝓛 += model.τ_2 * tr((v.W)' * v.D_2 * v.W)

        𝓛 += 0.5 * v.μ * norm(v.E .- v.Y .+ v.Q .- v.R .+ v.Λ./v.μ)^2.0
        𝓛 += 0.5 * v.μ * norm(v.Q .- bag_max!(v.Q_res, v.T, v.X_cut) .+ v.Σ./v.μ)^2.0
        𝓛 += 0.5 * v.μ * norm(v.T .- (v.WmX .+ v.b) .+ v.Θ./v.μ)^2.0
        𝓛 += 0.5 * v.μ * norm(v.R .- bag_max!(v.R_res, v.U, v.X_cut) .+ v.Ω./v.μ)^2.0
        𝓛 += 0.5 * v.μ * norm(v.U .- (v.WyX .+ v.by) .+ v.Ξ./v.μ)^2.0
        𝓛 += 0.5 * v.μ * norm(v.V .- v.W .+ v.Γ ./ v.μ)^2.0
    elseif target == "V"
        for g in 1:length(model.M_cut)
            V_g = v.V[model.M_cut[g], :]
            𝓛 += model.τ_1 * tr(V_g * v.D_1s[g] * V_g')
        end
        𝓛 += 0.5 * v.μ * norm(v.V .- v.W .+ v.Γ ./ v.μ)^2.0
    elseif target == "T"
        𝓛 += 0.5 * v.μ * norm(v.Q .- bag_max!(v.Q_res, v.T, v.X_cut) .+ v.Σ./v.μ)^2.0
        𝓛 += 0.5 * v.μ * norm(v.T .- (v.WmX .+ v.b) .+ v.Θ./v.μ)^2.0
    end        
    return 𝓛
end

function inexact_loss(model::MIMSSVMClassifier, v::mimmsvm_vars)
    d = size(v.W, 1)
    s = repeat(v.s', outer=(d, 1))

    newW = v.W - s.*v.∇W
    newWX = newW' * v.X
    K = size(v.Y, 1)
    newWyX = repeat(newWX[v.YI .> 0]', outer=(K, 1))

    𝓛 = 0.0
    𝓛 += 0.5 * norm(newW, 2)^2.0
    𝓛 += model.τ_2 * tr(newW' * v.D_2 * newW)
    𝓛 += 0.5 * v.μ * norm(v.V .- newW .+ v.Γ ./ v.μ)^2.0
    𝓛 += 0.5 * v.μ * norm(v.T .- (newWX .+ v.b) .+ v.Θ/v.μ)^2.0
    𝓛 += 0.5 * v.μ * norm(v.U .- (newWyX + v.by) .+ v.Ξ/v.μ)^2.0
end

function bag_max(WX, X_cut)
    return hcat([maximum(WX[:, cut], dims=2) for cut in X_cut]...)
end

function bag_max!(R, WX, X_cut)
    ## R = 1×416 Matrix{Float64}, 
    for (i, cut) in enumerate(X_cut)
        c = @view WX[:, cut] 
        r = @view R[:,i]
        maximum!(r, c)
    end
    return R
end

# function calc_precomputed_vars(v::mimmsvm_vars)
#     d, K = size(v.W)
# end

function calc_PDPs!(v::mimmsvm_vars)
    ## Not works in MMMISVM
    d, K = size(v.W)

    rhs1 = sum([v.X[:,cut]*v.X[:,cut]' for cut in v.X_cut]) 
    rhs2 = [zeros(d, d) for i in 1:K] 
    for m in 1:K
        step1 = [v.X[:,cut][:,v.YI[:,cut][m,:] .> 0] for cut in v.X_cut]
        rhs2[m] = sum([x * x' for x in step1])
    end
    As = [rhs1 + K*r2 for r2 in rhs2] 
    eigAs = [eigen(A) for A in As] 

    v.Ps = [real(eigA.vectors) for eigA in eigAs] 
    v.Ds = [Diagonal(real(eigA.values)) for eigA in eigAs] 
    v.P⁻¹s = [real(inv(eigA.vectors)) for eigA in eigAs] 

    v.Xᵀprime = [v.X[:,cut]' for cut in v.prime_cut] 
end

function calc_WmX_WyX!(v::mimmsvm_vars)
    mul!(v.WmX, v.W', v.X) 
    K = size(v.Y, 1) 
    v.tmpKNI .= v.WmX .* (v.YI .> 0) 
    for m in 1:K
        tmp = @view v.WyX[m:m,:] 
        sum!(tmp, v.tmpKNI)
    end
end

function calc_by!(v::mimmsvm_vars)
    for m in 1:size(v.Y, 1)
        bym = @view v.by[:, v.prime_cut[m]] 
        bym .= v.b[m] 
    end
end

function W_update!(model::MIMSSVMClassifier, v::mimmsvm_vars)
    ## refer: https://math.stackexchange.com/questions/670649/efficient-diagonal-update-of-matrix-inverse
    K, NI = size(v.T)
    d = size(v.W, 1)

    @. v.T_res = v.T - v.b + v.Θ/v.μ; T̂ = @view v.T_res[:,:] 
    ## "@." converts all the operations of that line into element-wise: https://stackoverflow.com/questions/65160733/what-does-an-at-sign-mean-in-julia
    @. v.U_res = v.U - v.by + v.Ξ/v.μ; Û = @view v.U_res[:,:] 
    mul!(v.tmpKd, T̂, v.X') 

    d, K = size(v.W)
    rhs1 = sum([v.X[:,cut]*v.X[:,cut]' for cut in v.X_cut]) 

    for m in 1:K
        Ûprime = @view Û[:,v.prime_cut[m]]; t̂Xᵀ = @view v.tmpKd[m,:]; w = @view v.W[:,m:m]

        mul!(v.tmpKd2, Ûprime, v.Xᵀprime[m]) 

        sum!(v.tmp1d, v.tmpKd2) 
        @. v.tmp1d += t̂Xᵀ' + (v.V[:, m])' + (v.Γ[:, m])' ./ v.μ 

        step1 = [v.X[:,cut][:,v.YI[:,cut][m,:] .> 0] for cut in v.X_cut]
        rhs2 = sum([x * x' for x in step1]) 

        As = rhs1 + K * rhs2 
        As += (2 / v.μ) * model.τ_2 * v.D_2 + (1.0 / v.μ + 1) * I
        
        v.W[:,m] = v.tmp1d * inv(As)
    end

    calc_WmX_WyX!(v)
end

function inexact_W_update!(model::MIMSSVMClassifier, v::mimmsvm_vars)
    s_update!(model::MIMSSVMClassifier, v::mimmsvm_vars)
    d = size(v.W, 1)

    @. v.W -= v.∇W .* v.s'

    calc_WmX_WyX!(v)
end

function s_update!(model::MIMSSVMClassifier, v::mimmsvm_vars)
    K, NI = size(v.T)

    v.T_res .= v.T .- v.WmX .- v.b .+ v.Θ./v.μ; T̂ = @view v.T_res[:,:]
    v.U_res .= v.U .- v.WyX .- v.by .+ v.Ξ./v.μ; Û = @view v.U_res[:,:]
    mul!(v.tmpKd, T̂,  v.X')
    v.tmpKd .= v.W' .- v.μ .* v.tmpKd 

    for m in 1:K
        Ûprime = @view Û[:,v.prime_cut[m]] 
        tmp1d = @view v.tmpKd[m,:] 
        w = @view v.W[:,m]
        mul!(v.tmpKprime[m], Ûprime, v.Xᵀprime[m]) 
        ÛXᵀprime = @view v.tmpKprime[m][:,:]

        ∇w = @view v.∇W[:,m] 
        sum!(∇w, ÛXᵀprime')

        ∇w .= tmp1d .- v.μ .* ∇w 
        ∇w += (- v.μ * (v.V[:, m] - w + v.Γ[:, m] / v.μ) + model.τ_2 * v.D_2 * w) 

        Xᵀ∇w = @view v.tmpKNI[m,:]; mul!(Xᵀ∇w, v.X', ∇w); t̂ = @view T̂[m,:]
        numer = w' * ∇w .- v.μ * (t̂' * Xᵀ∇w) .- v.μ * sum(ÛXᵀprime * ∇w, dims=1) 
        numer[1] += (w' * (2 * model.τ_2 .* v.D_2) - v.μ * (v.V[:, m] - w + v.Γ[:, m] ./ v.μ)') * ∇w 
        mul!(tmp1d, v.XXᵀ_KXXᵀprime[m], ∇w)
        denom = ∇w' * ∇w .+ v.μ .* ∇w' * tmp1d
        denom[1] +=  ∇w' * (v.μ * I + 2 * model.τ_2 * v.D_2) * ∇w
        v.s[m] = numer[1] / denom[1]
    end
end

function D_1s_update!(model::MIMSSVMClassifier, v::mimmsvm_vars)
    G = length(model.M_cut)
    for g in 1:G
        v_g = v.V[model.M_cut[g], :]
        for k in 1:v.K
            v.D_1s[g][k, k] = 0.5 * sqrt(norm(v_g[:, k], 2)^2 + model.δ)
        end
    end
end

function D_2_update!(model::MIMSSVMClassifier, v::mimmsvm_vars)
    # @infiltrate cond = any(isnan, v.W)
    v.D_2 = 0.5 * p_exponent_matrix(v.W * v.W' + model.δ * I, -0.5)
end

function b_update!(model::MIMSSVMClassifier, v::mimmsvm_vars)
    K, NI = size(v.YI)

    @. v.T_res = v.T - v.WmX + v.Θ/v.μ; T̂ = @view v.T_res[:,:]
    @. v.U_res = v.U - v.WyX + v.Ξ/v.μ; Û = @view v.U_res[:,:]
    sum!(v.b, T̂)
    for m in 1:K
        Ûprime = @view Û[:,v.prime_cut[m]]
        v.b[m] += sum(Ûprime)
    end
    v.b .= v.b ./ (NI .+ K .* v.prime_sum)

    calc_by!(v)
end

function E_update!(model::MIMSSVMClassifier, v::mimmsvm_vars)
    S = @view v.E_res[:,:]; S .= v.Y .- v.Q .+ v.R .- v.Λ./v.μ
    YS = @view v.Q_res[:,:]; YS .= v.Y .* S
    gt = YS .> model.C/v.μ
    mid = 0 .<= YS .<= model.C/v.μ

    v.E .= S .* .!mid .- gt .* v.Y .* (model.C/v.μ)
end

function Q_update!(model::MIMSSVMClassifier, v::mimmsvm_vars)
    bag_max!(v.Q_res, v.T, v.X_cut); bag_max_T = @view v.Q_res[:,:]
    v.Q .= 0.5 .* (v.Y .- v.E .+ v.R .- v.Λ./v.μ .+ bag_max_T .- v.Σ./v.μ)
end

function R_update!(model::MIMSSVMClassifier, v::mimmsvm_vars)
    bag_max!(v.R_res, v.U, v.X_cut); bag_max_U = @view v.R_res[:,:]
    v.R .= 0.5 .* (v.E .- v.Y .+ v.Q .+ v.Λ./v.μ .+ bag_max_U .- v.Ω./v.μ)
end

function T_update!(model::MIMSSVMClassifier, v::mimmsvm_vars)
    K = size(v.Y, 1)
    v.T_res .= v.WmX .+ v.b .- v.Θ./v.μ # Store data in T_res to save allocations
    Φ = @view v.T_res[:,:]
    for (i, cut) in enumerate(v.X_cut)
        ni = length(cut)
        for m in 1:K
            ϕᵢₘ = @view Φ[m, cut]
            v.T[m,cut] = ϕᵢₘ
            v.T[m,cut[1]+argmax(ϕᵢₘ)-1] = 0.5 * (maximum(ϕᵢₘ) + v.Q[m, i] + v.Σ[m, i]/v.μ)
        end
    end
    """
    for (i, cut) in enumerate(v.X_cut)
        ni = length(cut)
        for m in 1:K
            ϕᵢₘ_prime = sort(ϕᵢₘ, rev=true)
            a_imc = [(sum(ϕᵢₘ_prime[1:j]) + qᵢ[m] + σᵢ[m]/v.μ) / (j + 1) for j in 1:ni]
            #println(a_imc .> ϕᵢₘ_prime)
            c_star = argmax(a_imc .> ϕᵢₘ_prime) - 1
            #println(c_star)
            if c_star > 0
                a_imcstar = a_imc[c_star]
                v.T[m,cut] = min.(ϕᵢₘ, a_imcstar)
            else
                v.T[m,cut] = ϕᵢₘ
                v.T[m,cut[1]+argmax(ϕᵢₘ)-1]=a_imc[1]
            end
        end
    end
    """
end

function U_update!(model::MIMSSVMClassifier, v::mimmsvm_vars)
    K = size(v.Y, 1)
    v.U_res .= v.WyX .+ v.by .- v.Ξ./v.μ # Store data in U_res to save allocations
    Ψ = @view v.U_res[:,:]
    for (i, cut) in enumerate(v.X_cut)
        ni = length(cut)
        for m in 1:K
            ψᵢₘ = @view Ψ[m, cut]
            v.U[m,cut] = ψᵢₘ
            v.U[m,cut[1]+argmax(ψᵢₘ)-1] = 0.5 * (maximum(ψᵢₘ) + v.R[m, i] + v.Ω[m, i]/v.μ)
        end
    end
    """
    for (i, cut) in enumerate(v.X_cut)
        ni = length(cut)
        for m in 1:K
            ψᵢₘ_prime = sort(ψᵢₘ, rev=true)
            a_imc = [(sum(ψᵢₘ_prime[1:j]) + v.R[m,i] + v.Ω[m,i]/v.μ) / (j + 1) for j in 1:ni]
            #println(a_imc .> ψᵢₘ_prime)
            c_star = argmax(a_imc .> ψᵢₘ_prime)-1
            #println(c_star)
            if c_star > 1
                a_imcstar = a_imc[c_star]
                v.U[m,cut] = min.(ψᵢₘ, a_imcstar)
            else
                v.U[m,cut] = ψᵢₘ
                v.U[m,cut[1]+argmax(ψᵢₘ)-1] = a_imc[1]
            end
        end
    end
    """
end

function V_update!(model::MIMSSVMClassifier, v::mimmsvm_vars)
    G = length(model.M_cut)
    for g in 1:G
        right_parenthis = (2 * model.τ_1 .* v.D_1s[g] + v.μ .* I)
        diagindices = diagind(right_parenthis)
        right_parenthis[diagindices] = 1.0 ./ right_parenthis[diagindices] ## more efficient than matrix inversion.
        v.V[model.M_cut[g], :] = (v.μ .* v.W[model.M_cut[g], :] - v.Γ[model.M_cut[g], :]) * right_parenthis
    end
end

function calc_residuals!(model::MIMSSVMClassifier, v::mimmsvm_vars)
    v.E_res .= v.E .- (v.Y .- v.Q .+ v.R)
    v.Q_res .= v.Q .- bag_max!(v.Q_res, v.T, v.X_cut)
    v.T_res .= v.T .- (v.WmX .+ v.b)
    v.R_res .= v.R .- bag_max!(v.R_res, v.U, v.X_cut)
    v.U_res .= v.U .- (v.WyX .+ v.by)
end

function MIMSSVMClassifier(; M_cut=missing, δ=1e-10, τ_1=1.0, τ_2=1.0, C=1.0, μ=1e-3, ρ=1.2, maxiter=1000, tol=1e-6, use_CUDA=false, exact=true)
    @assert all(i -> (i > 0), [C, μ, ρ, maxiter, tol])
    @assert ρ > 1.0
    model = MIMSSVMClassifier(M_cut, δ, τ_1, τ_2, C, μ, ρ, maxiter, tol, use_CUDA, exact)
end

function MLJBase.fit(model::MIMSSVMClassifier, verbosity::Integer, X, y)
    v = init_vars(model, X, y)

    @assert !any(isnan, v.X)
    if verbosity > 5
        calc_residuals!(model, v)
        res = sum([norm(r) for r in (v.E_res, v.Q_res, v.T_res, v.R_res, v.U_res)])

        ol = obj_loss(model, v)
        ll = lagrangian_loss(model, v)
        print("Loss: " * string(ol) * "     \t") 
        print("Lagrangian: " * string(ll) * "     \t") 
        println("Residual: " * string(res))
    end

    #reset_timer!()
    for i in 1:model.maxiter
        D_1s_update!(model, v)
        D_2_update!(model, v)
        if model.exact
            W_update!(model, v)
        else
            inexact_W_update!(model, v)
        end
        b_update!(model, v)
        E_update!(model, v)
        Q_update!(model, v)
        R_update!(model, v)
        T_update!(model, v)
        U_update!(model, v)
        V_update!(model, v)

        calc_residuals!(model, v)

        @. v.Λ += v.μ * v.E_res
        @. v.Σ += v.μ * v.Q_res
        @. v.Θ += v.μ * v.T_res 
        @. v.Ω += v.μ * v.R_res
        @. v.Ξ += v.μ * v.U_res
        @. v.Γ += v.μ * v.V_res
         
        res = sum([norm(r) for r in (v.E_res, v.Q_res, v.T_res, v.R_res, v.U_res, v.V_res)])

        if verbosity > 5
            ol = obj_loss(model, v)
            ll = lagrangian_loss(model, v)
            print("Loss: " * string(ol) * "     \t")
            print("Lagrangian: " * string(ll) * "     \t")
            println("Residual: " * string(res))
        end

        if res < model.tol
            break
        end

        v.μ = model.ρ * v.μ
    end

    #if verbosity > 5
    #    print_timer()
    #end

    fitresult = v.W, v.b, levels(y)
    cache = missing
    report = missing

    return fitresult, cache, report
end

function MLJBase.predict(model::MIMSSVMClassifier, fitresult, Xnew)
    N = length(Xnew)
    W, b, levels_of_y = fitresult

    X = hcat([MLJBase.matrix(x)' for x in Xnew]...)
    nis = [size(x, 1) for x in Xnew]
    X_cut = [sum(nis[1:ni])-nis[ni]+1:sum(nis[1:ni]) for ni in 1:N]
    raw_pred = bag_max(W' * X .+ b, X_cut)

    pred = CategoricalArray(onecold(raw_pred, levels_of_y))

    return pred
end
