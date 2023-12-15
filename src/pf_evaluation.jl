"""
Wrapper for using QuantumClifford's naive_encoding_circuit function
"""
function evaluate_code_decoder_w_ecirc_pf(checks, ecirc, scirc, p_init, p_shift; nframes=10_000, encoding_locs=nothing)
    s, n = size(checks)
    if p_shift != 0
        non_mz, mz = clifford_grouper(scirc)
        non_mz = calculate_shifts(non_mz)
        scirc = []

        first_shift = true
        for subcircuit in non_mz
            # Shift!
            if !first_shift
                append!(scirc, [PauliError(i,p_shift) for i in n+1:n+s])
            end
            append!(scirc, subcircuit)
            first_shift = false
        end
        append!(scirc, mz)
    end

    return QuantumClifford.ECC.naive_error_correction_pipeline(checks, p_init, ecirc=ecirc, scirc=scirc, nframes=nframes, encoding_locs=encoding_locs)
end

"""Evaluates lookup table decoder for shor style syndrome circuit. Wrapper for ['shor_error_correction_pipeline'](@ref)
If no p_shift is provided, this runs as if it there were no shift errors. Some parameters:
*  P_init = probability of an initial error after encoding
*  P_shift = probability that a shift induces an error - was less than 0.01 in "Shuttling an Electron Spin through a Silicon Quantum Dot Array"
*  P_wait = probability that waiting causes a qubit to decohere
"""
function evaluate_code_decoder_shor_syndrome(checks::Stabilizer, ecirc, cat, scirc, p_init, p_shift=0, p_wait=0; nframes=10_000, encoding_locs=nothing)
    s, n = size(checks)
    anc_qubits = 0
    for pauli in checks
        anc_qubits += mapreduce(count_ones,+, xview(pauli) .| zview(pauli))
    end

    if p_shift != 0
        non_mz, mz = clifford_grouper(scirc)
        non_mz = calculate_shifts(non_mz)
        scirc = []

        first_shift = true
        for subcircuit in non_mz
            # Shift!
            if !first_shift
                # Errors due to shifting the data/ancilla row - whichever is smallest
                # TODO right now hardcoded to shift the data qubits.
                #append!(scirc, [PauliError(i,p_shift) for i in n+1:n+anc_qubits])
                append!(scirc, [PauliError(i,p_shift) for i in 1:n])
            end
            append!(scirc, subcircuit)
            first_shift = false

            # Errors due to waiting for the next shuttle -> should this be on all qubits? Maybe the p_shift includes this for ancilla already?
            # TODO Should this be random Pauli error or just Z error?
            #append!(scirc, [PauliError(i,p_wait) for i in 1:n])
            append!(scirc, [PauliError(i,p_wait) for i in n+1:n+anc_qubits])
        end
        append!(scirc, mz)
    end
    return QuantumClifford.ECC.shor_error_correction_pipeline(checks, p_init, cat=cat, scirc=scirc, nframes=nframes, ecirc=ecirc, encoding_locs=encoding_locs)
end

# This might be mathematically wrong. In practicy I've treated p_error = 1 - gate_fidelity
"""Takes a circuit and adds a pf noise op after each two qubit gate, correspond to the provided error probability."""
function add_two_qubit_gate_noise(circuit, p_error)
    new_circuit = []
    for gate in circuit
        if isa(gate, QuantumClifford.AbstractTwoQubitOperator)
            push!(new_circuit, gate)
            push!(new_circuit, PauliError(gate.q1, p_error))
            push!(new_circuit, PauliError(gate.q2, p_error))
        else
            push!(new_circuit, gate)
        end
    end
    return new_circuit
end

"""Same as [`vary_shift_errors_plot`](@ref) but uses pauli frame simulation"""
function vary_shift_errors_plot_pf(code::AbstractECC, name=string(typeof(code)))
    checks = parity_checks(code)
    vary_shift_errors_plot_pf(checks, name)
end
function vary_shift_errors_plot_pf(checks::Stabilizer, name="")
    title = name*" Circuit w/ Encoding Circuit PF"
    scirc, _ = naive_syndrome_circuit(checks)
    ecirc = naive_encoding_circuit(checks)

    error_rates = 0.000:0.00150:0.12
    # Uncompiled errors
    post_ec_error_rates_s0 = [evaluate_code_decoder_w_ecirc_pf(checks, ecirc, scirc, p, 0) for p in error_rates]
    post_ec_error_rates_s10 = [evaluate_code_decoder_w_ecirc_pf(checks, ecirc, scirc, p, p/10) for p in error_rates]
    post_ec_error_rates_s100 = [evaluate_code_decoder_w_ecirc_pf(checks, ecirc, scirc, p, p) for p in error_rates]
    x_error_s0 = [post_ec_error_rates_s0[i][1] for i in eachindex(post_ec_error_rates_s0)]
    z_error_s0 = [post_ec_error_rates_s0[i][2] for i in eachindex(post_ec_error_rates_s0)]
    x_error_s10 = [post_ec_error_rates_s10[i][1] for i in eachindex(post_ec_error_rates_s10)]
    z_error_s10 = [post_ec_error_rates_s10[i][2] for i in eachindex(post_ec_error_rates_s10)]
    x_error_s100 = [post_ec_error_rates_s100[i][1] for i in eachindex(post_ec_error_rates_s100)]
    z_error_s100 = [post_ec_error_rates_s100[i][2] for i in eachindex(post_ec_error_rates_s100)]

    # Anc Compiled circuit
    new_circuit, order = ancil_reindex_pipeline(scirc)
    compiled_post_ec_error_rates_s0 = [evaluate_code_decoder_w_ecirc_pf(checks, ecirc, new_circuit, p, 0) for p in error_rates]
    compiled_post_ec_error_rates_s10 = [evaluate_code_decoder_w_ecirc_pf(checks, ecirc, new_circuit, p, p/10) for p in error_rates]
    compiled_post_ec_error_rates_s100 = [evaluate_code_decoder_w_ecirc_pf(checks, ecirc, new_circuit, p, p) for p in error_rates]
    compiled_x_error_s0 = [compiled_post_ec_error_rates_s0[i][1] for i in eachindex(compiled_post_ec_error_rates_s0)]
    compiled_z_error_s0 = [compiled_post_ec_error_rates_s0[i][2] for i in eachindex(compiled_post_ec_error_rates_s0)]
    compiled_x_error_s10 = [compiled_post_ec_error_rates_s10[i][1] for i in eachindex(compiled_post_ec_error_rates_s10)]
    compiled_z_error_s10 = [compiled_post_ec_error_rates_s10[i][2] for i in eachindex(compiled_post_ec_error_rates_s10)]
    compiled_x_error_s100 = [compiled_post_ec_error_rates_s100[i][1] for i in eachindex(compiled_post_ec_error_rates_s100)]
    compiled_z_error_s100 = [compiled_post_ec_error_rates_s100[i][2] for i in eachindex(compiled_post_ec_error_rates_s100)]

    # Data + Anc Compiled circuit
    s, n = size(checks)
    k = n-s

    renewed_circuit, data_order = data_ancil_reindex(scirc, n+s)
    encoding_locs = []
    for i in n-k+1:n
        push!(encoding_locs, data_order[i])
    end
    renewed_ecirc = perfect_reindex(ecirc, data_order)

    dataQubits = n
    reverse_dict = Dict(value => key for (key, value) in data_order)
    parity_reindex = [reverse_dict[i] for i in collect(1:dataQubits)]
    renewed_checks = checks[:,parity_reindex]
    full_compiled_post_ec_error_rates_s0 = [evaluate_code_decoder_w_ecirc_pf(renewed_checks, renewed_ecirc, renewed_circuit, p, 0, encoding_locs=encoding_locs) for p in error_rates]
    full_compiled_post_ec_error_rates_s10 = [evaluate_code_decoder_w_ecirc_pf(renewed_checks, renewed_ecirc, renewed_circuit, p, p/10,  encoding_locs=encoding_locs) for p in error_rates]
    full_compiled_post_ec_error_rates_s100 = [evaluate_code_decoder_w_ecirc_pf(renewed_checks, renewed_ecirc, renewed_circuit, p, p,  encoding_locs=encoding_locs) for p in error_rates]
    full_compiled_x_error_s0 = [full_compiled_post_ec_error_rates_s0[i][1] for i in eachindex(full_compiled_post_ec_error_rates_s0)]
    full_compiled_z_error_s0 = [full_compiled_post_ec_error_rates_s0[i][2] for i in eachindex(full_compiled_post_ec_error_rates_s0)]
    full_compiled_x_error_s10 = [full_compiled_post_ec_error_rates_s10[i][1] for i in eachindex(full_compiled_post_ec_error_rates_s10)]
    full_compiled_z_error_s10 = [full_compiled_post_ec_error_rates_s10[i][2] for i in eachindex(full_compiled_post_ec_error_rates_s10)]
    full_compiled_x_error_s100 = [full_compiled_post_ec_error_rates_s100[i][1] for i in eachindex(full_compiled_post_ec_error_rates_s100)]
    full_compiled_z_error_s100 = [full_compiled_post_ec_error_rates_s100[i][2] for i in eachindex(full_compiled_post_ec_error_rates_s100)]

    f_x = Figure(resolution=(1100,900))
    ax = f_x[1,1] = Axis(f_x, xlabel="single (qu)bit error rate",title=title*" Logical X")
    lim = max(error_rates[end])
    lines!([0,lim], [0,lim], label="single bit", color=:black)

    # Uncompiled Plots
    scatter!(error_rates, x_error_s0, label="Original circuit with no shift errors", color=:red, marker=:circle)
    scatter!(error_rates, x_error_s10, label="Original circuit with shift errors = p/10", color=:red, marker=:utriangle)
    scatter!(error_rates, x_error_s100, label="Original circuit with shift errors = p", color=:red, marker=:star8)

    # Compiled Plots
    scatter!(error_rates, compiled_x_error_s0, label="Anc compiled circuit with no shift errors", color=:blue, marker=:circle)
    scatter!(error_rates, compiled_x_error_s10, label="Anc compiled circuit with shift errors = p/10", color=:blue, marker=:utriangle)
    scatter!(error_rates, compiled_x_error_s100, label="Anc compiled circuit with shift errors = p", color=:blue, marker=:star8)

    # Compiled Plots
     scatter!(error_rates, full_compiled_x_error_s0, label="Data + anc compiled circuit with no shift errors", color=:green, marker=:circle)
    scatter!(error_rates, full_compiled_x_error_s10, label="Data + anc compiled circuit with shift errors = p/10", color=:green, marker=:utriangle)
    scatter!(error_rates, full_compiled_x_error_s100, label="Data + anc compiled circuit with shift errors = p", color=:green, marker=:star8)

    f_x[1,2] = Legend(f_x, ax, "Error Rates")

    f_z = Figure(resolution=(1100,900))
    ax = f_z[1,1] = Axis(f_z, xlabel="single (qu)bit error rate",title=title*" Logical Z")
    lim = max(error_rates[end])
    lines!([0,lim], [0,lim], label="single bit", color=:black)

    # Uncompiled Plots
    scatter!(error_rates, z_error_s0, label="Original circuit with no shift errors", color=:red, marker=:circle)
    scatter!(error_rates, z_error_s10, label="Original circuit with shift errors = p/10", color=:red, marker=:utriangle)
    scatter!(error_rates, z_error_s100, label="Original circuit with shift errors = p", color=:red, marker=:star8)

    # Compiled Plots
    scatter!(error_rates, compiled_z_error_s0, label="Anc compiled circuit with no shift errors", color=:blue, marker=:circle)
    scatter!(error_rates, compiled_z_error_s10, label="Anc compiled circuit with shift errors = p/10", color=:blue, marker=:utriangle)
    scatter!(error_rates, compiled_z_error_s100, label="Anc compiled circuit with shift errors = p", color=:blue, marker=:star8)

     # Compiled Plots
    scatter!(error_rates, full_compiled_z_error_s0, label="Data + anc compiled circuit with no shift errors", color=:green, marker=:circle)
    scatter!(error_rates, full_compiled_z_error_s10, label="Data + anc compiled circuit with shift errors = p/10", color=:green, marker=:utriangle)
    scatter!(error_rates, full_compiled_z_error_s100, label="Data + anc compiled circuit with shift errors = p", color=:green, marker=:star8)

    f_z[1,2] = Legend(f_z, ax, "Error Rates")
    return f_x, f_z
end

"""Same as [`vary_shift_errors_plot_pf`](@ref) but uses fault tolerant syndrome measurement"""
function vary_shift_errors_plot_shor_syndrome(code::AbstractECC, name=string(typeof(code)))
    title = name*" Circuit - Shor Syndrome Circuit"
    checks = parity_checks(code)
    cat, scirc, anc_qubits, bit_indices = shor_syndrome_circuit(checks)
    ecirc = naive_encoding_circuit(code)

    constraints, data_qubits = size(checks)
    total_qubits = anc_qubits + data_qubits

    error_rates = 0.000:0.00150:0.12
    # Uncompiled errors
    post_ec_error_rates_s0 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, cat, scirc, p, 0) for p in error_rates]
    post_ec_error_rates_s10 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, cat, scirc, p, p/10) for p in error_rates]
    post_ec_error_rates_s100 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, cat, scirc, p, p) for p in error_rates]
    x_error_s0 = [post_ec_error_rates_s0[i][1] for i in eachindex(post_ec_error_rates_s0)]
    z_error_s0 = [post_ec_error_rates_s0[i][2] for i in eachindex(post_ec_error_rates_s0)]
    x_error_s10 = [post_ec_error_rates_s10[i][1] for i in eachindex(post_ec_error_rates_s10)]
    z_error_s10 = [post_ec_error_rates_s10[i][2] for i in eachindex(post_ec_error_rates_s10)]
    x_error_s100 = [post_ec_error_rates_s100[i][1] for i in eachindex(post_ec_error_rates_s100)]
    z_error_s100 = [post_ec_error_rates_s100[i][2] for i in eachindex(post_ec_error_rates_s100)]

    # Anc Compiled circuit
    new_circuit, order = ancil_reindex_pipeline(scirc)
    new_cat = perfect_reindex(cat, order)
    compiled_post_ec_error_rates_s0 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, new_cat, new_circuit, p, 0) for p in error_rates]
    compiled_post_ec_error_rates_s10 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, new_cat, new_circuit, p, p/10) for p in error_rates]
    compiled_post_ec_error_rates_s100 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, new_cat, new_circuit, p, p) for p in error_rates]
    compiled_x_error_s0 = [compiled_post_ec_error_rates_s0[i][1] for i in eachindex(compiled_post_ec_error_rates_s0)]
    compiled_z_error_s0 = [compiled_post_ec_error_rates_s0[i][2] for i in eachindex(compiled_post_ec_error_rates_s0)]
    compiled_x_error_s10 = [compiled_post_ec_error_rates_s10[i][1] for i in eachindex(compiled_post_ec_error_rates_s10)]
    compiled_z_error_s10 = [compiled_post_ec_error_rates_s10[i][2] for i in eachindex(compiled_post_ec_error_rates_s10)]
    compiled_x_error_s100 = [compiled_post_ec_error_rates_s100[i][1] for i in eachindex(compiled_post_ec_error_rates_s100)]
    compiled_z_error_s100 = [compiled_post_ec_error_rates_s100[i][2] for i in eachindex(compiled_post_ec_error_rates_s100)]

    # Data + Anc Compiled circuit
    renewed_circuit, data_order = data_ancil_reindex(scirc, total_qubits)
    renewed_ecirc = perfect_reindex(ecirc, data_order)
    renewed_cat = perfect_reindex(cat, data_order)

    # Data + Anc Compiled circuit
    s, n = size(checks)
    k = n-s

    encoding_locs = []
    for i in n-k+1:n
        push!(encoding_locs, data_order[i])
    end

    reverse_dict = Dict(value => key for (key, value) in data_order)
    parity_reindex = [reverse_dict[i] for i in collect(1:data_qubits)]
    renewed_checks = checks[:,parity_reindex]
    full_compiled_post_ec_error_rates_s0 = [evaluate_code_decoder_shor_syndrome(renewed_checks, renewed_ecirc, renewed_cat, renewed_circuit, p, 0, encoding_locs = encoding_locs) for p in error_rates]
    full_compiled_post_ec_error_rates_s10 = [evaluate_code_decoder_shor_syndrome(renewed_checks, renewed_ecirc, renewed_cat, renewed_circuit, p, p/10, encoding_locs = encoding_locs) for p in error_rates]
    full_compiled_post_ec_error_rates_s100 = [evaluate_code_decoder_shor_syndrome(renewed_checks, renewed_ecirc, renewed_cat, renewed_circuit, p, p, encoding_locs = encoding_locs) for p in error_rates]
    full_compiled_x_error_s0 = [full_compiled_post_ec_error_rates_s0[i][1] for i in eachindex(full_compiled_post_ec_error_rates_s0)]
    full_compiled_z_error_s0 = [full_compiled_post_ec_error_rates_s0[i][2] for i in eachindex(full_compiled_post_ec_error_rates_s0)]
    full_compiled_x_error_s10 = [full_compiled_post_ec_error_rates_s10[i][1] for i in eachindex(full_compiled_post_ec_error_rates_s10)]
    full_compiled_z_error_s10 = [full_compiled_post_ec_error_rates_s10[i][2] for i in eachindex(full_compiled_post_ec_error_rates_s10)]
    full_compiled_x_error_s100 = [full_compiled_post_ec_error_rates_s100[i][1] for i in eachindex(full_compiled_post_ec_error_rates_s100)]
    full_compiled_z_error_s100 = [full_compiled_post_ec_error_rates_s100[i][2] for i in eachindex(full_compiled_post_ec_error_rates_s100)]

    f_x = Figure(resolution=(1100,900))
    ax = f_x[1,1] = Axis(f_x, xlabel="single (qu)bit error rate",title=title*" Logical X")
    lim = max(error_rates[end])
    lines!([0,lim], [0,lim], label="single bit", color=:black)

    # Uncompiled Plots
    scatter!(error_rates, x_error_s0, label="Original circuit with no shift errors", color=:red, marker=:circle)
    scatter!(error_rates, x_error_s10, label="Original circuit with shift errors = p/10", color=:red, marker=:utriangle)
    scatter!(error_rates, x_error_s100, label="Original circuit with shift errors = p", color=:red, marker=:star8)

    # Compiled Plots
    scatter!(error_rates, compiled_x_error_s0, label="Anc compiled circuit with no shift errors", color=:blue, marker=:circle)
    scatter!(error_rates, compiled_x_error_s10, label="Anc compiled circuit with shift errors = p/10", color=:blue, marker=:utriangle)
    scatter!(error_rates, compiled_x_error_s100, label="Anc compiled circuit with shift errors = p", color=:blue, marker=:star8)

    # Compiled Plots
    scatter!(error_rates, full_compiled_x_error_s0, label="Data + anc compiled circuit with no shift errors", color=:green, marker=:circle)
    scatter!(error_rates, full_compiled_x_error_s10, label="Data + anc compiled circuit with shift errors = p/10", color=:green, marker=:utriangle)
    scatter!(error_rates, full_compiled_x_error_s100, label="Data + anc compiled circuit with shift errors = p", color=:green, marker=:star8)

    f_x[1,2] = Legend(f_x, ax, "Error Rates")

    f_z = Figure(resolution=(1100,900))
    ax = f_z[1,1] = Axis(f_z, xlabel="single (qu)bit error rate",title=title*" Logical Z")
    lim = max(error_rates[end])
    lines!([0,lim], [0,lim], label="single bit", color=:black)

    # Uncompiled Plots
    scatter!(error_rates, z_error_s0, label="Original circuit with no shift errors", color=:red, marker=:circle)
    scatter!(error_rates, z_error_s10, label="Original circuit with shift errors = p/10", color=:red, marker=:utriangle)
    scatter!(error_rates, z_error_s100, label="Original circuit with shift errors = p", color=:red, marker=:star8)

    # Compiled Plots
    scatter!(error_rates, compiled_z_error_s0, label="Anc compiled circuit with no shift errors", color=:blue, marker=:circle)
    scatter!(error_rates, compiled_z_error_s10, label="Anc compiled circuit with shift errors = p/10", color=:blue, marker=:utriangle)
    scatter!(error_rates, compiled_z_error_s100, label="Anc compiled circuit with shift errors = p", color=:blue, marker=:star8)

    # Compiled Plots
    scatter!(error_rates, full_compiled_z_error_s0, label="Data + anc compiled circuit with no shift errors", color=:green, marker=:circle)
    scatter!(error_rates, full_compiled_z_error_s10, label="Data + anc compiled circuit with shift errors = p/10", color=:green, marker=:utriangle)
    scatter!(error_rates, full_compiled_z_error_s100, label="Data + anc compiled circuit with shift errors = p", color=:green, marker=:star8)

    f_z[1,2] = Legend(f_z, ax, "Error Rates")
    return f_x, f_z
end

"""Same as [`vary_shift_errors_plot_shor_syndrome`](@ref) but also applies realistic noise pararmeters related to:
- Error rate per shift
- Error rate due to waiting (decoherence)
- Two qubit gate fidelity
"""
function realistic_noise_logical_physical_error(code::AbstractECC, p_shift=0.01, p_wait=1-exp(-14.5/28_000), p_gate=1-0.995; name=string(typeof(code)))
    title = name*" Circuit - Shor Syndrome Circuit"
    checks = parity_checks(code)
    cat, scirc, anc_qubits, bit_indices = shor_syndrome_circuit(checks)
    ecirc = naive_encoding_circuit(code)
    nframes = 20_000
    m = 1/3

    error_rates = 0.000:0.00150:0.30
    # Uncompiled errors
    post_ec_error_rates_s0 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, cat, add_two_qubit_gate_noise(scirc, p_gate), p, p_shift, p_wait, nframes=nframes) for p in error_rates]
    post_ec_error_rates_s1 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, cat, add_two_qubit_gate_noise(scirc, p_gate*m), p, p_shift*m, 1-exp(-14.5*m/28_000), nframes=nframes) for p in error_rates]
    x_error_s0 = [post_ec_error_rates_s0[i][1] for i in eachindex(post_ec_error_rates_s0)]
    z_error_s0 = [post_ec_error_rates_s0[i][2] for i in eachindex(post_ec_error_rates_s0)]
    x_error_s1 = [post_ec_error_rates_s1[i][1] for i in eachindex(post_ec_error_rates_s1)]
    z_error_s1 = [post_ec_error_rates_s1[i][2] for i in eachindex(post_ec_error_rates_s1)]

    # Anc Compiled circuit
    new_circuit, order = ancil_reindex_pipeline(scirc)
    new_cat = perfect_reindex(cat, order)
    compiled_post_ec_error_rates_s0 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, new_cat, add_two_qubit_gate_noise(new_circuit, p_gate), p, p_shift, p_wait, nframes=nframes) for p in error_rates]
    compiled_post_ec_error_rates_s1 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, new_cat, add_two_qubit_gate_noise(new_circuit, p_gate*m), p, p_shift*m, 1-exp(-14.5*m/28_000), nframes=nframes) for p in error_rates]
    compiled_x_error_s0 = [compiled_post_ec_error_rates_s0[i][1] for i in eachindex(compiled_post_ec_error_rates_s0)]
    compiled_z_error_s0 = [compiled_post_ec_error_rates_s0[i][2] for i in eachindex(compiled_post_ec_error_rates_s0)]
    compiled_x_error_s1 = [compiled_post_ec_error_rates_s1[i][1] for i in eachindex(compiled_post_ec_error_rates_s1)]
    compiled_z_error_s1 = [compiled_post_ec_error_rates_s1[i][2] for i in eachindex(compiled_post_ec_error_rates_s1)]

    f_x = Figure(resolution=(1100,900))
    ax = f_x[1,1] = Axis(f_x, xlabel="single (qu)bit error rate - AKA error after encoding",ylabel="Logical error rate",title=title*" Logical X")
    lim = max(error_rates[end])
    lines!([0,lim], [0,lim], label="single bit", color=:black)

    # Uncompiled Plots
    scatter!(error_rates, x_error_s0, label="Uncompiled with realistic errors", color=:red, marker=:star8)
    scatter!(error_rates, x_error_s1, label="Uncompiled w/ moderately optimistic (m=1/3) params", color=:red, marker=:utriangle)

    # Compiled Plots
    scatter!(error_rates, compiled_x_error_s0, label="Compiled (AC + GS) with realistic errors", color=:blue, marker=:star8)
    scatter!(error_rates, compiled_x_error_s1, label="Compiled w/ moderately optimistic (m=1/3) params", color=:blue, marker=:utriangle)

    f_x[1,2] = Legend(f_x, ax, "Error Rates")

    f_z = Figure(resolution=(1100,900))
    ax = f_z[1,1] = Axis(f_z, xlabel="single (qu)bit error rate - AKA error after encoding",ylabel="Logical error rate",title=title*" Logical Z")
    lim = max(error_rates[end])
    lines!([0,lim], [0,lim], label="single bit", color=:black)

    # Uncompiled Plots
    scatter!(error_rates, z_error_s0, label="Uncompiled with realistic errors", color=:red, marker=:star8)
    scatter!(error_rates, z_error_s1, label="Uncompiled w/ moderately optimistic (m=1/3) params", color=:red, marker=:utriangle)

    # Compiled Plots
    scatter!(error_rates, compiled_z_error_s0, label="Compiled (AC + GS) with realistic errors", color=:blue, marker=:star8)
    scatter!(error_rates, compiled_z_error_s1, label="Compiled w/ moderately optimistic (m=1/3) params", color=:blue, marker=:utriangle)

    f_z[1,2] = Legend(f_z, ax, "Error Rates")
    return f_x, f_z
end

"""This plot realistic performance of the 2xn architecture varying a scalar of the various error probabilities"""
function realistic_noise_vary_params(code::AbstractECC, p_shift=0.01, p_wait=1-exp(-14.5/28_000), p_gate=1-0.995; name=string(typeof(code)))
    title = name*" Circuit - Varrying the magnitude of realisitic noise params"
    checks = parity_checks(code)
    cat, scirc, anc_qubits, bit_indices = shor_syndrome_circuit(checks)
    ecirc = naive_encoding_circuit(code)
    nframes = 10_000

    error_rates = 0.00:0.05:4
    # Uncompiled errors
    post_ec_error_rates_s0 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, cat, add_two_qubit_gate_noise(scirc, p_gate*m),
                                0, p_shift*m, p_wait*m, nframes=nframes) for m in error_rates]

    x_error_s0 = [post_ec_error_rates_s0[i][1] for i in eachindex(post_ec_error_rates_s0)]
    z_error_s0 = [post_ec_error_rates_s0[i][2] for i in eachindex(post_ec_error_rates_s0)]

    # Anc Compiled circuit
    new_circuit, order = ancil_reindex_pipeline(scirc)
    new_cat = perfect_reindex(cat, order)
    compiled_post_ec_error_rates_s0 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, new_cat,
                                add_two_qubit_gate_noise(new_circuit, p_gate*m), 0, p_shift*m, p_wait*m, nframes=nframes) for m in error_rates]

    compiled_x_error_s0 = [compiled_post_ec_error_rates_s0[i][1] for i in eachindex(compiled_post_ec_error_rates_s0)]
    compiled_z_error_s0 = [compiled_post_ec_error_rates_s0[i][2] for i in eachindex(compiled_post_ec_error_rates_s0)]

    # X plots
    f_x = Figure(resolution=(1100,900))
    ax = f_x[1,1] = Axis(f_x, xlabel="Scaling of SoA physical error parameters",ylabel="Logical error rate", title=title*" Logical X")
    lim = max(error_rates[end])
    lines!([0,lim], [0.5,0.5], label="50% logical error", color=:black)
    lines!([1,1], [0,0.5], label="2023 State of the art (SoA) parameters", linestyle=:dash, color=:black)

    scatter!(error_rates, x_error_s0, label="Uncompiled with realistic errors", color=:red, marker=:star8)
    scatter!(error_rates, compiled_x_error_s0, label="Compiled (AC + GS) with realistic errors", color=:blue, marker=:star8)
    f_x[1,2] = Legend(f_x, ax, "Error Rates")

    # Z plots
    f_z = Figure(resolution=(1100,900))
    ax = f_z[1,1] = Axis(f_z, xlabel="Scaling of SoA physical error parameters",ylabel="Logical error rate", title=title*" Logical Z")
    lim = max(error_rates[end])
    lines!([0,lim], [0.5,0.5], label="50% logical error", color=:black)
    lines!([1,1], [0,0.5], label="2023 State of the art (SoA) parameters", linestyle=:dash, color=:black)

    scatter!(error_rates, z_error_s0, label="Uncompiled with realistic errors", color=:red, marker=:star8)
    scatter!(error_rates, compiled_z_error_s0, label="Compiled (AC + GS) with realistic errors", color=:blue, marker=:star8)
    f_z[1,2] = Legend(f_z, ax, "Error Rates")
    return f_x, f_z
end


"""Wrapper for QuantumClifford.ECC.CSS_naive_error_correction_pipeline()
"""
function evaluate_code_decoder_FTecirc_pf_krishna(code::CSS, scirc, p_init, p_shift=0 ; nframes=1_000, max_iters = 25)
    s, n = size(code.tableau)
    _, _, r = canonicalize!(Base.copy(code.tableau), ranks=true)
    k = n - r

    if p_shift != 0
        non_mz, mz = clifford_grouper(scirc)
        non_mz = calculate_shifts(non_mz)
        scirc = []

        first_shift = true
        for subcircuit in non_mz
            # Shift!
            if !first_shift
                append!(scirc, [PauliError(i,p_shift) for i in n+1:n+s])
            end
            append!(circuit_Z, subcircuit)
            first_shift = false
        end
        append!(scirc, mz)
    end

    return QuantumClifford.ECC.CSS_naive_error_correction_pipeline(code, p_init, nframes=nframes, scirc=scirc, max_iters=max_iters) 
end

"""Wrapper for CSS_shor_error_correction_pipeline()
"""
function evaluate_code_FTencode_FTsynd_Krishna(code::CSS, cat, scirc, p_init, p_shift=0, p_wait=0; nframes=10_000, max_iters = 25)
    s, n = size(code.tableau)

    anc_qubits = 0
    for pauli in code.tableau
        anc_qubits += mapreduce(count_ones,+, xview(pauli) .| zview(pauli))
    end

    if p_shift != 0
        non_mz, mz = clifford_grouper(scirc)
        non_mz = calculate_shifts(non_mz)
        scirc = []

        first_shift = true
        for subcircuit in non_mz
            # Shift!
            if !first_shift
                # Errors due to shifting the data/ancilla row - whichever is smallest
                # TODO right now hardcoded to shift the data qubits.
                #append!(scirc, [PauliError(i,p_shift) for i in n+1:n+anc_qubits])
                append!(scirc, [PauliError(i,p_shift) for i in 1:n])
            end
            append!(scirc, subcircuit)
            first_shift = false

            # Errors due to waiting for the next shuttle -> should this be on all qubits? Maybe the p_shift includes this for ancilla already?
            # TODO Should this be random Pauli error or just Z error?
            #append!(scirc, [PauliError(i,p_wait) for i in 1:n])
            append!(scirc, [PauliError(i,p_wait) for i in n+1:n+anc_qubits])
        end
        append!(scirc, mz)
    end
    return QuantumClifford.ECC.CSS_shor_error_correction_pipeline(code, p_init, cat=cat, scirc=scirc, nframes=nframes, max_iters=max_iters)
end

# This had terrible results
"""Effects of compilation against varying memory error after encoding + other realistic sources of error""" 
function realistic_noise_logical_physical_error_ldpc(code::CSS, p_shift=0.01, p_wait=1-exp(-14.5/28_000), p_gate=1-0.995; name="")
    title = name*" Circuit - Shor Syndrome Circuit"

    checks = code.tableau
    cat, scirc, _ = shor_syndrome_circuit(checks)

    nframes = 1000
    m = 1/3
    p_wait = 0 # this term is fine
    p_gate = 0 # gate error alone causes terrible performance
    p_shift = p_shift/100

    error_rates = exp10.(range(-5,-1.5,length=40))
    # Uncompiled errors
    post_ec_error_rates_s0 = [evaluate_code_FTencode_FTsynd_Krishna(code, cat, add_two_qubit_gate_noise(scirc, p_gate), p, p_shift, p_wait, nframes=nframes) for p in error_rates]
    post_ec_error_rates_s1 = [evaluate_code_FTencode_FTsynd_Krishna(code, cat, add_two_qubit_gate_noise(scirc, p_gate*m), p, p_shift*m, 1-exp(-14.5*m/28_000), nframes=nframes) for p in error_rates]
    x_error_s0 = [post_ec_error_rates_s0[i][1] for i in eachindex(post_ec_error_rates_s0)]
    z_error_s0 = [post_ec_error_rates_s0[i][2] for i in eachindex(post_ec_error_rates_s0)]
    x_error_s1 = [post_ec_error_rates_s1[i][1] for i in eachindex(post_ec_error_rates_s1)]
    z_error_s1 = [post_ec_error_rates_s1[i][2] for i in eachindex(post_ec_error_rates_s1)]

    # Anc Compiled circuit
    new_circuit, order = ancil_reindex_pipeline(scirc)
    new_cat = perfect_reindex(cat, order)
    compiled_post_ec_error_rates_s0 = [evaluate_code_FTencode_FTsynd_Krishna(code, new_cat, add_two_qubit_gate_noise(new_circuit, p_gate), p, p_shift, p_wait, nframes=nframes) for p in error_rates]
    compiled_post_ec_error_rates_s1 = [evaluate_code_FTencode_FTsynd_Krishna(code, new_cat, add_two_qubit_gate_noise(new_circuit, p_gate*m), p, p_shift*m, 1-exp(-14.5*m/28_000), nframes=nframes) for p in error_rates]
    compiled_x_error_s0 = [compiled_post_ec_error_rates_s0[i][1] for i in eachindex(compiled_post_ec_error_rates_s0)]
    compiled_z_error_s0 = [compiled_post_ec_error_rates_s0[i][2] for i in eachindex(compiled_post_ec_error_rates_s0)]
    compiled_x_error_s1 = [compiled_post_ec_error_rates_s1[i][1] for i in eachindex(compiled_post_ec_error_rates_s1)]
    compiled_z_error_s1 = [compiled_post_ec_error_rates_s1[i][2] for i in eachindex(compiled_post_ec_error_rates_s1)]

    f_x = Figure(resolution=(1100,900))
    ax = f_x[1,1] = Axis(f_x, xlabel="Log10 single (qu)bit error rate - AKA error after encoding",ylabel="Log10 Logical error rate",title=title*" Logical X")

    lines!([-5, 0], [-5, 0], label="single bit", color=:black)

    # Uncompiled Plots
    scatter!(log10.(error_rates), log10.(x_error_s0), label="Uncompiled with realistic errors", color=:red, marker=:star8)
    scatter!(log10.(error_rates), log10.(x_error_s1), label="Uncompiled w/ moderately optimistic (m=1/3) params", color=:red, marker=:utriangle)

    # Compiled Plots
    scatter!(log10.(error_rates), log10.(compiled_x_error_s0), label="Compiled (AC + GS) with realistic errors", color=:blue, marker=:star8)
    scatter!(log10.(error_rates), log10.(compiled_x_error_s1), label="Compiled w/ moderately optimistic (m=1/3) params", color=:blue, marker=:utriangle)

    f_x[1,2] = Legend(f_x, ax, "Error Rates")

    f_z = Figure(resolution=(1100,900))
    ax = f_z[1,1] = Axis(f_z, xlabel="Log10 single (qu)bit error rate - AKA error after encoding",ylabel="Log10 Logical error rate",title=title*" Logical Z")

    lines!([-5, 0], [-5, 0], label="single bit", color=:black)

    # Uncompiled Plots
    scatter!(log10.(error_rates), log10.(z_error_s0), label="Uncompiled with realistic errors", color=:red, marker=:star8)
    scatter!(log10.(error_rates), log10.(z_error_s1), label="Uncompiled w/ moderately optimistic (m=1/3) params", color=:red, marker=:utriangle)

    # Compiled Plots
    scatter!(log10.(error_rates), log10.(compiled_z_error_s0), label="Compiled (AC + GS) with realistic errors", color=:blue, marker=:star8)
    scatter!(log10.(error_rates), log10.(compiled_z_error_s1), label="Compiled w/ moderately optimistic (m=1/3) params", color=:blue, marker=:utriangle)

    f_z[1,2] = Legend(f_z, ax, "Error Rates")
    return f_x, f_z
end
