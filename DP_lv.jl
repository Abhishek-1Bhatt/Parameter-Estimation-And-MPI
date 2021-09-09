using StaticArrays
using Plots
using BenchmarkTools

function f(u, p)
    α, β, γ, δ = p

    du1 = α*u[1] - β*u[1]*u[2]
    du2 = -γ*u[2] + δ*u[1]*u[2]

    @SVector [du1, du2]
end

function dopri5!(u_cache, u, p, t_span, f)
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

    @inbounds for i in 2:Int64(t_span/dt)
        k1 = f(u, p)
        k2 = f(u + dt.*(a21.*k1), p)
        k3 = f(u + dt.*(a31.*k1 + a32.*k2), p)
        k4 = f(u + dt.*(a41.*k1 + a42.*k2 + a43.*k3), p)
        k5 = f(u + dt.*(a51.*k1 + a52.*k2 + a53.*k3 + a54.*k4), p)
        k6 = f(u + dt.*(a61.*k1 + a62.*k2 + a63.*k3 + a64.*k4 + a65.*k5), p)
        u = u + dt.*(b1.*k1 + b3.*k3 + b4.*k4 + b5.*k5 + b6.*k6)
        u_cache[:, i] = u
    end
end

p = (1.5, 1.0, 3.0, 1.0)
u₀ = [1.0, 1.0]
u_cache = Array{Float64}(undef, 2, 40)
u_cache[:, 1] = u₀
t_span = 10.0
u_cache

@btime dopri5!(u_cache, u₀, p, t_span, f)

dopri5!(u_cache, u₀, p, t_span, f)

u_cache

plot(0.25:0.25:t_span, u_cache[1, :])
plot!(0.25:0.25:t_span, u_cache[2, :])
