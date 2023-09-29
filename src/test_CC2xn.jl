using CircuitCompilation2xn
using QuantumClifford
using QuantumClifford.ECC: Steane7, Shor9, naive_syndrome_circuit, shor_syndrome_circuit, parity_checks, code_s, code_n, code_k
using CairoMakie
using Random
using Statistics
using Distributions
using NPZ
using QuantumClifford.ECC: naive_encoding_circuit, Cleve8, AbstractECC

function test_code(code)
    ecirc = encoding_circuit(code)
    mcirc, _ = naive_syndrome_circuit(code)

    new_circuit, order = CircuitCompilation2xn.ancil_reindex_pipeline(mcirc)

    diff = CircuitCompilation2xn.evaluate(mcirc, new_circuit, ecirc, code_n(code), code_s(code), code_s(code))

    println("\nNumber of discrepancies between the reordered circuit and the original over all possible 1 qubit Pauli errors inserted right after the encoding circuit:")
    println(sum(diff))
end

function test_full_reindex(code)
    ecirc = encoding_circuit(code)
    mcirc, _ = naive_syndrome_circuit(code)

    new_circuit, data_order = CircuitCompilation2xn.data_ancil_reindex(code)

    # Reindex encoding circuit
    new_ecirc = CircuitCompilation2xn.perfect_reindex(ecirc, data_order)

    diff = CircuitCompilation2xn.evaluate(mcirc, new_circuit, ecirc, code_n(code), code_s(code), code_s(code), new_ecirc, data_order)

    println("\nNumber of discrepancies between the reordered circuit and the original over all possible 1 qubit Pauli errors inserted right after the encoding circuit:")
    println(sum(diff))
    return new_ecirc, new_circuit
end

function test_full_reindex_plot(code, name=string(typeof(code)))
    ecirc = encoding_circuit(code)
    new_circuit, data_order = CircuitCompilation2xn.data_ancil_reindex(code)

    # Reindex encoding circuit
    new_ecirc = CircuitCompilation2xn.perfect_reindex(ecirc, data_order)

    error_rates = 0.000:0.0025:0.08
    H = parity_checks(code)
    dataQubits = size(H)[2]
    reverse_dict = Dict(value => key for (key, value) in data_order)
    parity_reindex = [reverse_dict[i] for i in collect(1:dataQubits)]
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc(H[:,parity_reindex], new_ecirc, new_circuit, p) for p in error_rates]
    f1 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Data + Anc Reindexed "*name*" w/ Encoding Circuit")
    return f1
end

function no_encoding_plot(code::Stabilizer, name=string(typeof(code)))
    scirc, _ = naive_syndrome_circuit(code)

    error_rates = 0.000:0.0025:0.08
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder(code, scirc, p) for p in error_rates]
    f1 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Original "*name*" Circuit - Syndrome Circuit")

    new_circuit, order = CircuitCompilation2xn.ancil_reindex_pipeline(scirc)
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder(code, new_circuit, p) for p in error_rates]
    f2 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Reordered "*name*" Circuit - Syndrome Circuit")
    return f1, f2
end

function no_encoding_plot(code, name=string(typeof(code)))
    scirc, _ = naive_syndrome_circuit(code)

    error_rates = 0.000:0.0025:0.08
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder(parity_checks(code), scirc, p) for p in error_rates]
    f1 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title=""*name*" Circuit - No Encoding")

    new_circuit, order = CircuitCompilation2xn.ancil_reindex_pipeline(scirc)
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder(parity_checks(code), new_circuit, p) for p in error_rates]
    f2 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Reordered "*name*" Circuit - Syndrome Circuit")
    return f1, f2
end

function encoding_plot(code, name=string(typeof(code)))
    checks = parity_checks(code)
    scirc, _ = naive_syndrome_circuit(code)
    ecirc = encoding_circuit(code)

    error_rates = 0.000:0.0025:0.08
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc(checks, ecirc, scirc, p) for p in error_rates]
    f1 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Original "*name*" Circuit w/ Encoding Circuit")

    new_circuit, order = CircuitCompilation2xn.ancil_reindex_pipeline(scirc)
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc(checks, ecirc, new_circuit, p) for p in error_rates]
    f2 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Reordered "*name*" Circuit w/ Encoding Circuit")
    return f1, f2
end

function pf_encoding_plot(code::AbstractECC, name=string(typeof(code)))
    checks = parity_checks(code)
    pf_encoding_plot(checks, name)
end

function pf_encoding_plot(checks, name="")
    reduced_checks = copy(stabilizerview(MixedDestabilizer(checks)))
    scirc, _ = naive_syndrome_circuit(checks)
    ecirc = naive_encoding_circuit(reduced_checks)

    error_rates = 0.000:0.0025:0.08
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc_pf(checks, ecirc, scirc, p, 0) for p in error_rates]
    x_error = [post_ec_error_rates[i][1] for i in eachindex(post_ec_error_rates)]
    z_error = [post_ec_error_rates[i][2] for i in eachindex(post_ec_error_rates)]

    f_x = CircuitCompilation2xn.plot_code_performance(error_rates, x_error,title="Logical X Error of "*name*" Circuit PF")
    f_z = CircuitCompilation2xn.plot_code_performance(error_rates, z_error,title="Logical Z Error of "*name*" Circuit PF")

    return f_x, f_z
    
    # Data-anc compile the circuit
    s, n = size(checks)
    k = n-s
    new_circuit, data_order = CircuitCompilation2xn.data_ancil_reindex(scirc, s+n)

    # Calculate locations of encoding qubits
    encoding_locs = []
    for i in n-k+1:n
        push!(encoding_locs, data_order[i])
    end

    # Reindex encoding circuit
    new_ecirc = CircuitCompilation2xn.perfect_reindex(ecirc, data_order)

    # Reindex the parity checks via checks[:,parity_reindex]
    dataQubits = n
    reverse_dict = Dict(value => key for (key, value) in data_order)
    parity_reindex = [reverse_dict[i] for i in collect(1:dataQubits)]

    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc_pf(checks[:,parity_reindex], new_ecirc, new_circuit, p, 0, encoding_locs=encoding_locs) for p in error_rates]
    x_error = [post_ec_error_rates[i][1] for i in eachindex(post_ec_error_rates)]
    z_error = [post_ec_error_rates[i][2] for i in eachindex(post_ec_error_rates)]

    new_f_x = CircuitCompilation2xn.plot_code_performance(error_rates, x_error,title="Logical X Error of "*name*" Circuit PF")
    new_f_z = CircuitCompilation2xn.plot_code_performance(error_rates, z_error,title="Logical Z Error of "*name*" Circuit PF")

    #return new_f_x, new_f_z
end

function encoding_plot_shifts(code, name=string(typeof(code)))
    scirc, _ = naive_syndrome_circuit(code)
    ecirc = encoding_circuit(code)

    error_rates = 0.000:0.00150:0.08
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc_shifts(parity_checks(code), ecirc, scirc, p, p/10) for p in error_rates]

    new_circuit, order = CircuitCompilation2xn.ancil_reindex_pipeline(scirc)
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

    return [mean(raw_shifts), mean(gate_shuffling_shifts),  mean(final_shifts)]
end

function test_LDPC_shift_reduction_shor_syndrome(n,k,w_r, samples=5)
    raw_shifts = []
    gate_shuffling_shifts = []
    final_shifts = []
    for i in 1:samples
        #matrix  = generate_LDPC_code(n,k,w_r)
        matrix = rand(Bernoulli(w_r/n), n-k, n) #using this as an approximation
        stab = Stabilizer(zeros(Bool,n-k, n), matrix)
        cat, scirc, anc_qubits, bit_indices = shor_syndrome_circuit(stab)

        circuit_wo_mz, measurement_circuit = CircuitCompilation2xn.clifford_grouper(scirc)
        push!(raw_shifts, length(CircuitCompilation2xn.calculate_shifts(circuit_wo_mz)))
        
        push!(gate_shuffling_shifts, length((CircuitCompilation2xn.gate_Shuffle(circuit_wo_mz))))
        
        constraints, data_qubits = size(stab)
        total_qubits = anc_qubits+data_qubits

        # TODO confirm that data ancil reindexing works as expected for these generated LDPC codes
        final_circ, order = CircuitCompilation2xn.data_ancil_reindex(circuit_wo_mz, total_qubits)
        push!(final_shifts, length(CircuitCompilation2xn.calculate_shifts(final_circ)))
    end

    println("\nRow weight: ", w_r)
    println("Raw shifts: ", mean(raw_shifts))
    println("After gate shuffling shifts: ", mean(gate_shuffling_shifts))
    println("After data-ancil reindexing shifts: ", mean(final_shifts))

    return [mean(raw_shifts), mean(gate_shuffling_shifts),  mean(final_shifts)]
end

function average_cooc(n,k,w_r, samples=5)
    raw_shifts = []
    gate_shuffling_shifts = []
    final_shifts = []
    for i in 1:samples
        #matrix  = generate_LDPC_code(n,k,w_r)
        matrix = rand(Bernoulli(w_r/n), n-k, n) #using this as an approximation
        stab = Stabilizer(zeros(Bool,n-k, n), matrix)
        scirc, _ = naive_syndrome_circuit(stab)
        
        circuit_wo_mz, measurement_circuit = CircuitCompilation2xn.clifford_grouper(scirc)
        numGates = length(circuit_wo_mz)
        push!(raw_shifts, numGates/length(CircuitCompilation2xn.calculate_shifts(circuit_wo_mz)))
        
        push!(gate_shuffling_shifts, numGates/length((CircuitCompilation2xn.gate_Shuffle(circuit_wo_mz))))
        
        # TODO confirm that data ancil reindexing works as expected for these generated LDPC codes
        final_circ, order = CircuitCompilation2xn.data_ancil_reindex(circuit_wo_mz, 2n-k)
        push!(final_shifts, numGates/length(CircuitCompilation2xn.calculate_shifts(final_circ)))
    end

    println("\nRow weight: ", w_r)
    println("Raw co-oc: ", mean(raw_shifts))
    println("After gate shuffling co-oc: ", mean(gate_shuffling_shifts))
    println("After data-ancil reindexing co-oc: ", mean(final_shifts))

    return [mean(raw_shifts), mean(gate_shuffling_shifts),  mean(final_shifts)]
end

function plot_LDPC_shift_reduction_ratio(n=100)
    row_weights = 5:15

    k = Int.([0.05n, 0.10n, 0.20n])
    
    a = [test_LDPC_shift_reduction(n,k[1],w_r) for w_r in row_weights]
    a = reduce(hcat, a)

    b = [test_LDPC_shift_reduction(n,k[2],w_r) for w_r in row_weights]
    b = reduce(hcat, b)

    c = [test_LDPC_shift_reduction(n,k[3],w_r) for w_r in row_weights]
    c = reduce(hcat, c)

    f = Figure(resolution=(1100,900))
    ax = f[1,1] = Axis(f, xlabel="Row Weights", ylabel="Ratio of Number of Uncompiled Shifts over Compiled Shifts", title="N= "*string(n)*" Random Classical LDPC Codes Shift Reductions")

    # Compiled Plots
    lines!(row_weights, a[2,:]./a[1,:], label="Gate Shuffling; r = 5%", color=:blue, linestyle = nothing)
    lines!(row_weights, b[2,:]./b[1,:], label="Gate Shuffling; r = 10%", color=:blue, linestyle = :dot)
    lines!(row_weights, c[2,:]./c[1,:], label="Gate Shuffling; r = 20%", color=:blue, linestyle = :dash)

    # Compiled Plots
    lines!(row_weights, a[3,:]./a[1,:], label="Data anc reordering; r = 5%", color=:green, linestyle = nothing)
    lines!(row_weights, b[3,:]./b[1,:], label="Data anc reordering; r = 10%", color=:green, linestyle = :dot)
    lines!(row_weights, c[3,:]./c[1,:], label="Data anc reordering; r = 20%", color=:green, linestyle = :dash)
    
    f[1,2] = Legend(f, ax, "Error Rates")
    
    return f
end

function plot_LDPC_shift_reduction_shiftPcheck(n=1000)
    row_weights = 5:15

    k = Int.([0.05n, 0.10n, 0.20n])
    
    a = [test_LDPC_shift_reduction(n,k[1],w_r) for w_r in row_weights]
    a = reduce(hcat, a)/(n-k[1])

    b = [test_LDPC_shift_reduction(n,k[2],w_r) for w_r in row_weights]
    b = reduce(hcat, b)/(n-k[2])

    c = [test_LDPC_shift_reduction(n,k[3],w_r) for w_r in row_weights]
    c = reduce(hcat, c)/(n-k[3])

    f = Figure(resolution=(1100,900))
    ax = f[1,1] = Axis(f, xlabel="Row Weights", ylabel="Average number of shifts per check", title="N= "*string(n)*" Random Classical LDPC Codes Shift Reductions")

    # Uncompiled Plots
    lines!(row_weights, a[1,:], label="No compilation; r = 5%", color=:red, linestyle = nothing)
    lines!(row_weights, b[1,:], label="No compilation; r = 10%", color=:red, linestyle = :dot)
    lines!(row_weights, c[1,:], label="No compilation; r = 20%", color=:red, linestyle = :dash)

    # Compiled Plots
    lines!(row_weights, a[2,:], label="Gate Shuffling; r = 5%", color=:blue, linestyle = nothing)
    lines!(row_weights, b[2,:], label="Gate Shuffling; r = 10%", color=:blue, linestyle = :dot)
    lines!(row_weights, c[2,:], label="Gate Shuffling; r = 20%", color=:blue, linestyle = :dash)

    # Compiled Plots
    lines!(row_weights, a[3,:], label="Data anc reordering; r = 5%", color=:green, linestyle = nothing)
    lines!(row_weights, b[3,:], label="Data anc reordering; r = 10%", color=:green, linestyle = :dot)
    lines!(row_weights, c[3,:], label="Data anc reordering; r = 20%", color=:green, linestyle = :dash)
    
    f[1,2] = Legend(f, ax, "Error Rates")
    
    return f
end

function plot_LDPC_shift_reduction_cooc(n=100)
    row_weights = 5:15

    k = Int.([0.05n, 0.10n, 0.20n])
    
    a = [average_cooc(n,k[1],w_r) for w_r in row_weights]
    a = reduce(hcat, a)

    b = [average_cooc(n,k[2],w_r) for w_r in row_weights]
    b = reduce(hcat, b)

    c = [average_cooc(n,k[3],w_r) for w_r in row_weights]
    c = reduce(hcat, c)

    f = Figure(resolution=(1100,900))
    ax = f[1,1] = Axis(f, xlabel="Row Weights", ylabel="Average number of parallel gates per shuffle", title="N= "*string(n)*" Random Classical LDPC Codes Shift Reductions")

    # Uncompiled Plots
    lines!(row_weights, a[1,:], label="No compilation; r = 5%", color=:red, linestyle = nothing)
    lines!(row_weights, b[1,:], label="No compilation; r = 10%", color=:red, linestyle = :dot)
    lines!(row_weights, c[1,:], label="No compilation; r = 20%", color=:red, linestyle = :dash)

    # Compiled Plots
    lines!(row_weights, a[2,:], label="Gate Shuffling; r = 5%", color=:blue, linestyle = nothing)
    lines!(row_weights, b[2,:], label="Gate Shuffling; r = 10%", color=:blue, linestyle = :dot)
    lines!(row_weights, c[2,:], label="Gate Shuffling; r = 20%", color=:blue, linestyle = :dash)

    # Compiled Plots
    lines!(row_weights, a[3,:], label="Data anc reordering; r = 5%", color=:green, linestyle = nothing)
    lines!(row_weights, b[3,:], label="Data anc reordering; r = 10%", color=:green, linestyle = :dot)
    lines!(row_weights, c[3,:], label="Data anc reordering; r = 20%", color=:green, linestyle = :dash)
    
    f[1,2] = Legend(f, ax, "Error Rates")
    
    return f
end

function test_shor_circuit_reindexing(code, name=string(typeof(code)))
    checks = parity_checks(code)
    cat, scirc, anc_qubits, bit_indices = shor_syndrome_circuit(checks)
    ecirc = encoding_circuit(code)

    error_rates = 0.000:0.0025:0.08

    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_shor_syndrome(checks, ecirc, cat, scirc, p, 0) for p in error_rates]
    x_error = [post_ec_error_rates[i][1] for i in eachindex(post_ec_error_rates)]
    z_error = [post_ec_error_rates[i][2] for i in eachindex(post_ec_error_rates)]

    f_x = CircuitCompilation2xn.plot_code_performance(error_rates, x_error,title="Logical X Error of "*name*" Circuit Shor_Syndrome")
    f_z = CircuitCompilation2xn.plot_code_performance(error_rates, z_error,title="Logical Z Error of "*name*" Circuit Shor_Syndrome")
    
    # anc compile the circuit
    new_circuit, order = CircuitCompilation2xn.ancil_reindex_pipeline(scirc)
    new_cat = CircuitCompilation2xn.perfect_reindex(cat,order)

    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_shor_syndrome(checks, ecirc, new_cat, new_circuit, p, 0) for p in error_rates]
    x_error = [post_ec_error_rates[i][1] for i in eachindex(post_ec_error_rates)]
    z_error = [post_ec_error_rates[i][2] for i in eachindex(post_ec_error_rates)]

    new_f_x = CircuitCompilation2xn.plot_code_performance(error_rates, x_error,title="Logical X Error of AncReindex"*name*" Circuit ShorSynd")
    new_f_z = CircuitCompilation2xn.plot_code_performance(error_rates, z_error,title="Logical Z Error of AncReindex"*name*" Circuit ShorSynd")
    
    #return f_x, f_z  
    
    return new_f_x, new_f_z
end

function stab_from_cxcz(Cx, Cz)
    num_x_checks, qubits = size(Cx)
    num_z_checks, qubits2 = size(Cz)
    # qubits and qubits2 should be the same value
    stab = Stabilizer(vcat(zeros(Bool,num_z_checks,qubits),Cx), vcat(Cz, zeros(Bool,num_x_checks,qubits2)) )
    return stab
end

function getGoodLDPC(n=1)
    stab = nothing
    # Absolute paths to Cx and Cz npz files:
    if n==1
        Cx = npzread("/Users/micciche/Research/QuantumInfo23/JuliaProjects/codes_for_hardware_test/1_ra1_rb2_X_rankX120_rankZ179_minWtX2_minWtZ2.npz");
        Cz = npzread("/Users/micciche/Research/QuantumInfo23/JuliaProjects/codes_for_hardware_test/1_ra1_rb2_Z_rankX120_rankZ179_minWtX2_minWtZ2.npz");
        stab = stab_from_cxcz(Cx,Cz);
    elseif n==2
        Cx = npzread("/Users/micciche/Research/QuantumInfo23/JuliaProjects/codes_for_hardware_test/1_ra2_rb2_X_rankX226_rankZ120_minWtX2_minWtZ2.npz");
        Cz = npzread("/Users/micciche/Research/QuantumInfo23/JuliaProjects/codes_for_hardware_test/1_ra2_rb2_Z_rankX226_rankZ120_minWtX2_minWtZ2.npz");
        stab = stab_from_cxcz(Cx,Cz);
    elseif n==3
        Cx = npzread("/Users/micciche/Research/QuantumInfo23/JuliaProjects/codes_for_hardware_test/3_ra1_rb2_X_rankX120_rankZ179_minWtX2_minWtZ2.npz");
        Cz = npzread("/Users/micciche/Research/QuantumInfo23/JuliaProjects/codes_for_hardware_test/3_ra1_rb2_Z_rankX120_rankZ179_minWtX2_minWtZ2.npz");
        stab = stab_from_cxcz(Cx,Cz);
    end
    return stab
end

function real_LDPC_numbers()
    println("First one")
    stab = getGoodLDPC(1);
    data_qubits = size(stab)[2]
    scirc, anc_bits, _ = naive_syndrome_circuit(stab)
    total_qubits = anc_bits+data_qubits

    println("Naive syndrome:", CircuitCompilation2xn.comp_numbers(scirc, total_qubits))
    cat, scirc, anc_bits, _ = shor_syndrome_circuit(stab)
    total_qubits = anc_bits+data_qubits
    println("Shor syndrome:", CircuitCompilation2xn.comp_numbers(scirc, total_qubits))

    println("\nSecond one")
    stab = getGoodLDPC(2);
    data_qubits = size(stab)[2]
    scirc, anc_bits, _ = naive_syndrome_circuit(stab)
    total_qubits = anc_bits+data_qubits

    println("Naive syndrome:", CircuitCompilation2xn.comp_numbers(scirc, total_qubits))
    cat, scirc, anc_bits, _ = shor_syndrome_circuit(stab)
    total_qubits = anc_bits+data_qubits
    println("Shor syndrome:", CircuitCompilation2xn.comp_numbers(scirc, total_qubits))

    println("\nThird one")
    stab = getGoodLDPC(3);
    data_qubits = size(stab)[2]
    scirc, anc_bits, _ = naive_syndrome_circuit(stab)
    total_qubits = anc_bits+data_qubits

    println("Naive syndrome:", CircuitCompilation2xn.comp_numbers(scirc, total_qubits))
    cat, scirc, anc_bits, _ = shor_syndrome_circuit(stab)
    total_qubits = anc_bits+data_qubits
    println("Shor syndrome:", CircuitCompilation2xn.comp_numbers(scirc, total_qubits))
end
#println("\n######################### Steane7 #########################")
#test_code(Steane7())

#println("\n######################### Shor9 #########################")
#test_code(Shor9())

#println("\n######################### Shor9 Plots #########################")

#orig, new = encoding_plot(Steane7())
#orig, new = encoding_plot(Shor9())
#orig, new = encoding_plot(Cleve8())

#f_x_Steane, f_z_Steane = pf_encoding_plot(Steane7())
#f_x_Shor, f_z_Shor = pf_encoding_plot(Shor9())
#f_x_Cleve, f_z_Cleve = pf_encoding_plot(Cleve8())
#f_x, f_z = pf_encoding_plot(getGoodLDPC(1), "LDPC1")
#f_x, f_z = pf_encoding_plot(getGoodLDPC(2), "LDPC2")
#f_x, f_z = pf_encoding_plot(getGoodLDPC(3), "LDPC3")

f_x, f_z = CircuitCompilation2xn.vary_shift_errors_plot_pf(Steane7())
#f_x, f_z = CircuitCompilation2xn.vary_shift_errors_plot_pf(Shor9())
#f_x, f_z = CircuitCompilation2xn.vary_shift_errors_plot_pf(Cleve8())

#plot_3 = encoding_plot_shifts(Steane7())
#plot_3 = encoding_plot_shifts(Shor9())

#steane_e, steane_s = test_full_reindex(Steane7())
#shor_e, shor_s = test_full_reindex(Shor9())

#test_full_reindex_plot(Shor9())

#plot = plot_LDPC_shift_reduction_shiftPcheck()
#plot = plot_LDPC_shift_reduction_cooc()

#f_x_Steane, f_z_Steane = test_shor_circuit_reindexing(Steane7())
#f_x_Shor, f_z_Shor = test_shor_circuit_reindexing(Shor9())

#f_x_Steane, f_z_Steane = CircuitCompilation2xn.vary_shift_errors_plot_shor_syndrome(Steane7())
#f_x_Shor, f_z_Shor = CircuitCompilation2xn.vary_shift_errors_plot_shor_syndrome(Shor9())
