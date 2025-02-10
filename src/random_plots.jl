using CairoMakie
using Statistics
using LDPCDecoders
using CircuitCompilation2xn
"""
* `n`: The length of the code i.e. number of bits/qubits
* `wr`: Row weight (Number of bits a parity check equations acts upon). Must divide n
* `wc`: Column weight (Number of parity check equations for a given bit)"""
function get_random_classic_code(n=120, wr=6, wc=8)
    H = LDPCDecoders.parity_check_matrix(n, wr, wc)

    checks = Stabilizer(zeros(Bool,size(H)),H)

    checks
end

function calculate_naive_average_shifts(n=120, wr=6, wc=8, samples=20)
    
    results = []
    for _ in 1:samples
        checks = get_random_classic_code(n, wr, wc)
        scirc, _ = QuantumClifford.ECC.naive_syndrome_circuit(checks)
        push!(results, CircuitCompilation2xn.comp_numbers(scirc, sum(size(checks)))[4])
    end

    return mean(results), std(results)
end

function calculate_shor_average_shifts(n=120, wr=6, wc=8, samples=20)
    
    results = []
    for _ in 1:samples
        checks = get_random_classic_code(n, wr, wc)
        cat, scirc, _ = QuantumClifford.ECC.shor_syndrome_circuit(checks)
        push!(results, CircuitCompilation2xn.shorNumbers(scirc)[4])
    end

    return mean(results), std(results)
end

function plot_rand_LDPC_vary_wr()
    row_weights = [1,2,3,4,5,6,8,10,12,15,24,30,40,60]
    naive_circ_means = [8.0
    289.7
    254.1
    211.8
    188.6
    172.2
    157.0
    146.2
    139.8
    133.85
    125.5
    123.05
    122.55
    121.8]

    naive_circ_stds = [ 0.0
    6.2500526313573435
    4.700503892361462
    6.645932669884007
    7.27215308738174
    6.152449234162551
    4.723959088908816
    3.981536333998142
    3.981536333998142
    2.3004576203785843
    1.9867985355975657
    2.3277502126799554
    1.4317821063276353
    1.7947291248483563]

    shor_circ_means = [8,8,8,8,8,8,8,8,8,8,8,8,8,8]

    pt = 4/3
    f = Figure(size=(500, 450),px_per_unit = 5.0, fontsize = 12.5pt)
    
    f_x =  f[1,1]
    ax = f[1,1] = Axis(f_x, xlabel="Row Weight",ylabel="Required Shuttles After Compilation",title="Shor Syndrome vs. Naive Syndrome Compilation for \nn=120 Random Classical LDPC codes with Wc= 8")
    scatter!(f_x, row_weights, shor_circ_means, label="Shor Syndrome ", color=:orange, marker=:diamond)
    errorbars!(row_weights, naive_circ_means, naive_circ_stds, color = :red)

    scatter!(f_x, row_weights, naive_circ_means, label="Naive Syndrome", color=:blue, marker=:circle)

    axislegend(ax)
    
    f
end

function plot_rand_LDPC_vary_wc()
    column_weights = [1,2,3,4,5,6,8,10,12,15,24,30,40,60]
    naive_circ_means = [8.0
    82.6
    98.4
   108.05
   120.0
   131.4
   158.1
   183.7
   211.2
   250.95
   375.5
   458.9
   605.7
   896.35]

    naive_circ_stds = [ 0.0
    4.070432538764349
    2.962928849706157
    4.454152409180733
    4.768316485434157
    6.816388918975856
    6.281970860704216
    5.7500572079534
    5.217379862687683
    5.808115918808421
    5.165676192553671
    3.193743884534262
    4.449719092257398
    3.199917762101164]

    shor_circ_means = column_weights

    pt = 4/3
    f = Figure(size=(500, 450),px_per_unit = 5.0, fontsize = 12.5pt)

    f_x =  f[1,1]
    ax = f[1,1] = Axis(f_x, xlabel="Column Weight",ylabel="Required Shuttles After Compilation",title="Shor Syndrome vs. Naive Syndrome Compilation for\nn=120 Random Classical LDPC codes with Wr= 8")
    scatter!(f_x, column_weights, shor_circ_means, label="Shor Syndrome", color=:orange, marker=:diamond)
    errorbars!(column_weights, naive_circ_means, naive_circ_stds, color = :red)

    scatter!(f_x, column_weights, naive_circ_means, label="Naive Syndrome", color=:blue, marker=:circle)

    axislegend(ax, halign=:left)
    f
end

function brickwork_plot(k, n, samples; nlayers=20)
    data = []
    max_col_w = []
    for _ in 1:samples
        code = random_brickwork_circuit_code((n,), nlayers, 1:(n/k|> Int):n)
        checks = parity_checks(code)

        cat, scirc, _ = QuantumClifford.ECC.shor_syndrome_circuit(checks)
        comp_numbers = CircuitCompilation2xn.shorNumbers(scirc)
        push!(data, comp_numbers)
        push!(max_col_w, CircuitCompilation2xn.max_column_weight(checks))
    end
    
    data = reduce(vcat,transpose.(data))

    # fig, ax, sp = series(data, labels=["Wc: "*string(max_col_w[i]) for i in 1:samples], xlabels = (["A", "B", "C", "D"]))
    
    # axislegend(ax)
    # return  fig
end