using CircuitCompilation2xn
using QuantumClifford
using QuantumClifford.ECC: Steane7, Shor9, naive_syndrome_circuit, encoding_circuit, parity_checks, code_s, code_n, code_k
using Quantikz
using CairoMakie
using Random
using Statistics
using Distributions

function test_code(code)
    ecirc = encoding_circuit(code)
    mcirc = naive_syndrome_circuit(code)

    new_circuit, order = CircuitCompilation2xn.ancil_reindex(mcirc)

    diff = CircuitCompilation2xn.evaluate(mcirc, new_circuit, ecirc, code_n(code), code_s(code), code_s(code))

    println("\nNumber of discrepancies between the reordered circuit and the original over all possible 1 qubit Pauli errors inserted right after the encoding circuit:")
    println(sum(diff))
end

function test_full_reindex(code)
    ecirc = encoding_circuit(code)
    mcirc = naive_syndrome_circuit(code)

    new_circuit, data_order = CircuitCompilation2xn.data_ancil_reindex(code)

    # Reindex encoding circuit
    new_ecirc = CircuitCompilation2xn.encoding_reindex(ecirc, data_order)

    diff = CircuitCompilation2xn.evaluate(mcirc, new_circuit, ecirc, code_n(code), code_s(code), code_s(code), new_ecirc, data_order)

    println("\nNumber of discrepancies between the reordered circuit and the original over all possible 1 qubit Pauli errors inserted right after the encoding circuit:")
    println(sum(diff))
    return new_ecirc, new_circuit
end

function test_full_reindex_plot(code, name=string(typeof(code)))
    ecirc = encoding_circuit(code)
    new_circuit, data_order = CircuitCompilation2xn.data_ancil_reindex(code)

    # Reindex encoding circuit
    new_ecirc = CircuitCompilation2xn.encoding_reindex(ecirc, data_order)

    error_rates = 0.000:0.0025:0.08
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc(parity_checks(code)[:,data_order], new_ecirc, new_circuit, p) for p in error_rates]
    f1 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Data + Anc Reindexed "*name*" w/ Encoding Circuit")
    return f1
end

function no_encoding_plot(code, name=string(typeof(code)))
    scirc = naive_syndrome_circuit(code)

    error_rates = 0.000:0.0025:0.08
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder(parity_checks(code), scirc, p) for p in error_rates]
    f1 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Original "*name*" Circuit - Syndrome Circuit")

    new_circuit, order = CircuitCompilation2xn.ancil_reindex(scirc)
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder(parity_checks(code), new_circuit, p) for p in error_rates]
    f2 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Reordered "*name*" Circuit - Syndrome Circuit")
    return f1, f2
end

function encoding_plot(code, name=string(typeof(code)))
    scirc = naive_syndrome_circuit(code)
    ecirc = encoding_circuit(code)

    error_rates = 0.000:0.0025:0.08
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc(parity_checks(code), ecirc, scirc, p) for p in error_rates]
    f1 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Original "*name*" Circuit w/ Encoding Circuit")

    new_circuit, order = CircuitCompilation2xn.ancil_reindex(scirc)
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc(parity_checks(code), ecirc, new_circuit, p) for p in error_rates]
    f2 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Reordered "*name*" Circuit w/ Encoding Circuit")
    return f1, f2
end

function pf_encoding_plot(code, name=string(typeof(code)))
    scirc = naive_syndrome_circuit(code)
    ecirc = encoding_circuit(code)

    error_rates = 0.000:0.0025:0.08
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc_pf(code, ecirc, scirc, p) for p in error_rates]
    f1 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Original "*name*" Circuit w/ Encoding Circuit PF")


    # TODO the pf decoder needs to reorder the logical measuring circuit
    #new_circuit, order = CircuitCompilation2xn.ancil_reindex(scirc)
    #post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc_pf(code, ecirc, new_circuit, p) for p in error_rates]
    #f2 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Reordered "*name*" Circuit w/ Encoding Circuit PF")
    #return f1,f2
end

function encoding_plot_shifts(code, name=string(typeof(code)))
    scirc = naive_syndrome_circuit(code)
    ecirc = encoding_circuit(code)

    error_rates = 0.000:0.00150:0.08
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc_shifts(parity_checks(code), ecirc, scirc, p, p/10) for p in error_rates]

    new_circuit, order = CircuitCompilation2xn.ancil_reindex(scirc)
    post_ec_error_rates_shifts = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc_shifts(parity_checks(code), ecirc, new_circuit, p, p/10) for p in error_rates]
    original = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc_shifts(parity_checks(code), ecirc, new_circuit, p, 0) for p in error_rates]
    plot = CircuitCompilation2xn.plot_code_performance_shift(error_rates, post_ec_error_rates, post_ec_error_rates_shifts,original, title=name*" Circuit w/ Encoding Circuit")
    return plot
end

function generate_LDPC_matrix(n, k, w_r)
    function generate_row(n,w_r)
        onePart = ones(Bool, w_r)
        zeroPart = zeros(Bool, n - w_r)
        return shuffle(append!(onePart,zeroPart))'
    end
    mat = generate_row(n, w_r-1)
    
    for i in 1:(n-k-1)
        mat = vcat(mat, generate_row(n, w_r))
    end

    return mat
end

function generate_LDPC_code(n,k,w_r)
    function check(matrix)
        redo = false
        for column in eachrow(matrix')
            if sum(column) < 2 #TODO Replace this 2 with a parameter maybe?
                redo = true
            end
        end
        return redo
    end

    matrix = generate_LDPC_matrix(n, k, w_r)

    # Throw out matrices with low column weights
    timeout = 0
    while check(matrix) && timeout<100
        matrix = generate_LDPC_matrix(n, k, w_r)
        timeout += 1
    end
    return matrix
end

function test_LDPC_shift_reduction(n,k,w_r, samples=5)
    raw_shifts = []
    gate_shuffling_shifts = []
    final_shifts = []
    for i in 1:samples
        #matrix  = generate_LDPC_code(n,k,w_r)
        matrix = rand(Bernoulli(w_r/n), n-k, n) #using this as an approximation
        stab = Stabilizer(zeros(Bool,n-k, n), matrix)
        scirc = naive_syndrome_circuit(stab)

        circuit_wo_mz, measurement_circuit = CircuitCompilation2xn.clifford_grouper(scirc)
        push!(raw_shifts, length(CircuitCompilation2xn.calculate_shifts(circuit_wo_mz)))
        
        push!(gate_shuffling_shifts, length((CircuitCompilation2xn.gate_Shuffle(circuit_wo_mz))))
        
        # TODO confirm that data ancil reindexing works as expected for these generated LDPC codes
        final_circ, order = CircuitCompilation2xn.data_ancil_reindex(circuit_wo_mz, 2n-k)
        push!(final_shifts, length(CircuitCompilation2xn.calculate_shifts(final_circ)))
    end

    println("\nRow weight: ", w_r)
    println("Raw shifts: ", mean(raw_shifts))
    println("After gate shuffling shifts: ", mean(gate_shuffling_shifts))
    println("After data-ancil reindexing shifts: ", mean(final_shifts))

    return  [mean(raw_shifts), mean(gate_shuffling_shifts),  mean(final_shifts)]
end
#println("\n######################### Steane7 #########################")
#test_code(Steane7())

#println("\n######################### Shor9 #########################")
#test_code(Shor9())

#println("\n######################### Shor9 Plots #########################")

#orig, new = encoding_plot(Steane7())
#orig, new = encoding_plot(Shor9())

#orig, new = pf_encoding_plot(Steane7())
#orig, new = pf_encoding_plot(Shor9())

#plot_3 = encoding_plot_shifts(Steane7())
#plot_3 = encoding_plot_shifts(Shor9())

#plot = CircuitCompilation2xn.vary_shift_errors_plot(Steane7())
#plot = CircuitCompilation2xn.vary_shift_errors_plot(Shor9())

steane_e, steane_s = test_full_reindex(Steane7())
shor_e, shor_s = test_full_reindex(Shor9())

#test_full_reindex_plot(Shor9())
function plot_LDPC_shift_reduction(n=100)
    row_weights = 5:12

    k = Int.([0.05n, 0.10n, 0.20n])
    
    a = [test_LDPC_shift_reduction(n,k[1],w_r) for w_r in row_weights]
    a = reduce(hcat, a)

    b = [test_LDPC_shift_reduction(n,k[2],w_r) for w_r in row_weights]
    b = reduce(hcat, b)

    c = [test_LDPC_shift_reduction(n,k[3],w_r) for w_r in row_weights]
    c = reduce(hcat, c)

    f = Figure(resolution=(1100,900))
    ax = f[1,1] = Axis(f, xlabel="Row Weights", ylabel="Number of Shifts to Run on 2xn", title="N=10,000 Random Classical LDPC Code Shift Reductions")

    # Uncompiled Plots
    #lines!(row_weights, a[1,:], label="No compilation; k = 5%", color=:red, linestyle = nothing)
    #lines!(row_weights, b[1,:], label="No compilation; k = 10%", color=:red, linestyle = :dot)
    #lines!(row_weights, c[1,:], label="No compilation; k = 20%", color=:red, linestyle = :dash)

    # Compiled Plots
    lines!(row_weights, a[2,:], label="Gate Shuffling; k = 5%", color=:blue, linestyle = nothing)
    lines!(row_weights, b[2,:], label="Gate Shuffling; k = 10%", color=:blue, linestyle = :dot)
    lines!(row_weights, c[2,:], label="Gate Shuffling; k = 20%", color=:blue, linestyle = :dash)

    # Compiled Plots
    lines!(row_weights, a[3,:], label="Data anc reordering; k = 5%", color=:green, linestyle = nothing)
    lines!(row_weights, b[3,:], label="Data anc reordering; k = 10%", color=:green, linestyle = :dot)
    lines!(row_weights, c[3,:], label="Data anc reordering; k = 20%", color=:green, linestyle = :dash)
    
    f[1,2] = Legend(f, ax, "Error Rates")
    
    return f
end

#a = plot_LDPC_shift_reduction()