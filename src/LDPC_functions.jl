function stab_from_cxcz(Cx, Cz)
    num_x_checks, qubits = size(Cx)
    num_z_checks, qubits2 = size(Cz)
    # qubits and qubits2 should be the same value
    stab = Stabilizer(vcat(Cx,zeros(Bool,num_z_checks,qubits)), vcat(zeros(Bool,num_x_checks,qubits2), Cz) )
    return stab
end

function getGoodLDPC(n=1)
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
    return stab, Cx, Cz
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
        matrix = rand(Distributions.Bernoulli(w_r/n), n-k, n) #using this as an approximation
        stab = Stabilizer(zeros(Bool,n-k, n), matrix)
        scirc, _ = naive_syndrome_circuit(stab)

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
        matrix = rand(Distributions.Bernoulli(w_r/n), n-k, n) #using this as an approximation
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
        matrix = rand(Distributions.Bernoulli(w_r/n), n-k, n) #using this as an approximation
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