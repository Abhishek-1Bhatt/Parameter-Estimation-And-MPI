using StaticArrays
using BenchmarkTools
using Statistics

mutable struct Multidual{N, T}
    val::T
    derivs::SVector{N, T}
end

Base.:+(f::Multidual, g::Multidual) = Multidual(f.val + g.val, f.derivs + g.derivs)
Base.:+(f::Multidual, g::Number) = Multidual(f.val + g, f.derivs)
Base.:+(f::Number, g::Multidual) = g + f

Base.:-(f::Multidual, g::Multidual) = Multidual(f.val - g.val, f.derivs - g.derivs)
Base.:-(f::Multidual, g::Number) = Multidual(f.val - g, f.derivs)
Base.:-(f::Number, g::Multidual) = g - f
Base.:-(f::Multidual) = Multidual(-f.val, -1 .* f.derivs)

Base.:*(f::Multidual, g::Multidual) = Multidual(f.val*g.val, f.val .* g.derivs + g.val .* f.derivs)
Base.:*(α::Number, f::Multidual) = Multidual(f.val * α, α .* f.derivs)
Base.:*(f::Multidual, α::Number) = α * f

function f(u, p)
    α, β, γ, δ = p

    du1 = α*u[1] - β*u[1]*u[2]
    du2 = -γ*u[2] + δ*u[1]*u[2]

    @SVector [du1, du2]
end

function dopri5!(u_vals, p, t_span, f, derivs)
    # Values from the Dormand Prince Tableau
    a21  =  (1.0/5.0)
    a31  = (3.0/40.0)
    a32  = (9.0/40.0)
    a41  = (44.0/45.0)
    a42  = (-56.0/15.0)
    a43  = (32.0/9.0)
    a51  = (19372.0/6561.0)
    a52  = (-25360.0/2187.0)
    a53  = (64448.0/6561.0)
    a54  = (-212.0/729.0)
    a61  = (9017.0/3168.0)
    a62  = (-355.0/33.0)
    a63  = (46732.0/5247.0)
    a64  = (49.0/176.0)
    a65  = (-5103.0/18656.0)
    a71  = (35.0/384.0)
    a72  = (0.0)
    a73  = (500.0/1113.0)
    a74  = (125.0/192.0)
    a75  = (-2187.0/6784.0)
    a76  = (11.0/84.0)

    b1   = (35.0/384.0)
    b2   = (0.0)
    b3   = (500.0/1113.0)
    b4   = (125.0/192.0)
    b5   = (-2187.0/6784.0)
    b6   = (11.0/84.0)
    b7   = (0.0)

    dt = 0.25
    uₚ = SVector{2, Float64}(u_vals[1, :])
    @inbounds for i in 1:Int64(t_span/dt)
        k1 = f(uₚ, p)
        k2 = f(uₚ + dt.*(a21.*k1), p)
        k3 = f(uₚ + dt.*(a31.*k1 + a32.*k2), p)
        k4 = f(uₚ + dt.*(a41.*k1 + a42.*k2 + a43.*k3), p)
        k5 = f(uₚ + dt.*(a51.*k1 + a52.*k2 + a53.*k3 + a54.*k4), p)
        k6 = f(uₚ + dt.*(a61.*k1 + a62.*k2 + a63.*k3 + a64.*k4 + a65.*k5), p)
        u = uₚ + dt.*(b1.*k1 + b3.*k3 + b4.*k4 + b5.*k5 + b6.*k6)
        u_vals[i, 1] = u[1].val
        u_vals[i, 2] = u[2].val
        derivs[i, 1:2] = u[1].derivs
        derivs[i, 3:4] = u[2].derivs
        uₚ = SVector{2, Float64}(u_vals[i, :])
    end
end

# Generating the dataset for parameter estimation
tp1 = Multidual(1.5, SVector(0.0, 0.0))
tp2 = Multidual(1.0, SVector(0.0, 0.0))
tp3 = Multidual(3.0, SVector(0.0, 0.0))
tp4 = Multidual(1.0, SVector(0.0, 0.0))

tp = (tp1, tp2, tp3, tp4)

dataset = Array{Float64}(undef, 40, 2)
dataset[1, :] = [1.0, 1.0]
t_derivs = Array{Float64}(undef, 40, 4)

t_span = 10.0
dopri5!(dataset, tp, t_span, f, t_derivs)


function cost_fun(derivs, ds, up, m)
    diff = abs.(ds .- up)
    dcdu = SVector{2, Float64}((1/m) .* sum(diff, dims=1))
    derivs[:, 1:2] = dcdu[1] .* derivs[:, 1:2] # dx/dp
    derivs[:, 3:4] = dcdu[2] .* derivs[:, 3:4] # dy/dp
    dcdp = SVector{4, Float64}(mean(derivs, dims=1))
    cost = SVector{2, Float64}((1/(2*m)) .* sum((diff .^ 2), dims=1))
    return @SVector[cost, dcdp]
end


function estim_params!(p, ds, t_span, f)
    m = 40
    up = Array{Float64}(undef, 40, 2)
    derivs = Array{Float64}(undef, 40, 4)
    cost = SVector{2, Float64}(1.0, 1.0)
    diff = 1.0
    tol = 10e-6
    lr = 0.001
    count = 0

    while !isapprox(diff, 0.0, atol=tol)
        up[1, :] = [1.0, 1.0]
        dopri5!(up, p, t_span, f, derivs)
        prev = cost
        cost, dcdp = cost_fun(derivs, ds, up, m)
        p[1].val = p[1].val - lr*dcdp[1]
        p[2].val = p[2].val - lr*dcdp[2]
        p[3].val = p[3].val - lr*dcdp[3]
        p[4].val = p[4].val - lr*dcdp[4]
        diff = sum(abs.(cost .- prev))
        count += 1
        if count%250 == 0
            println("Training pass $(count), cost = $(cost), $(p[1].val), $(p[2].val), $(p[3].val), $(p[4].val)")
        end
    end
    println("Finished Training.\nThe estimated parameters are: \nα = $(p[1].val)\ttrue value = 1.5\nβ = $(p[2].val)\ttrue value = 1.0\nγ = $(p[3].val)\ttrue value = 3.0\nδ = $(p[4].val)\ttrue value = 1.0\n")
end

α = Multidual(1.2, SVector(1.0, 0.0))
β = Multidual(0.8, SVector(0.0, 1.0))
γ = Multidual(2.8, SVector(1.0, 0.0))
δ = Multidual(0.8, SVector(0.0, 1.0))

p = (α, β, γ, δ)

estim_params!(p, dataset, t_span, f)
