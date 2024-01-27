"""
    Note used to be called `evaluate_code_decoder_w_ecirc_pf``
"""
# TODO add realistic noise capabilities to this like evaluate_code_decoder_shor_syndrome
function evaluate_code_decoder_naive(checks, decoder, ecirc, scirc, p_init, p_shift; nsamples=10_000)
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

    return naive_pipeline(checks, decoder, p_init, ecirc=ecirc, scirc=scirc, nsamples=nsamples)
end

"""Evaluates lookup table decoder for shor style syndrome circuit. Wrapper for ['shor_error_correction_pipeline'](@ref)
If no p_shift is provided, this runs as if it there were no shift errors. Some parameters:
*  P_init = probability of an initial error after encoding
*  P_shift = probability that a shift induces an error (per hop)- was less than 0.01% in "Shuttling an Electron Spin through a Silicon Quantum Dot Array"
*  P_wait = probability that waiting causes a qubit to decohere
"""
function evaluate_code_decoder_shor_syndrome(checks::Stabilizer, decoder::AbstractSyndromeDecoder, ecirc, cat, scirc, p_init, p_shift=0, p_wait=0; nsamples=10_000)
    s, n = size(checks)
    anc_qubits = 0
    for pauli in checks
        anc_qubits += mapreduce(count_ones,+, xview(pauli) .| zview(pauli))
    end

    if p_shift != 0 
        non_mz, mz = clifford_grouper(scirc)
        non_mz = calculate_shifts(non_mz)
        scirc = []
        
        current_delta = 0 # Starts with ancil qubits not lined up with any of the physical ones
        for subcircuit in non_mz
            # Shift!
            new_delta = abs(subcircuit[1].q2-subcircuit[1].q1)
            hops_to_new_delta = abs(new_delta-current_delta)
            hops_to_new_delta = 1 # TODO delte this line
            shift_error = p_shift * hops_to_new_delta

            append!(scirc, [PauliError(i,shift_error) for i in n+1:n+anc_qubits])
            
            # TODO Should this be random Pauli error or just Z error?
            append!(scirc, [PauliError(i,p_wait) for i in 1:n]) #TODO shoudl be random Z 
            append!(scirc, subcircuit)
        end
        append!(scirc, mz)
    end
    return shor_pipeline(checks, decoder, p_init, cat=cat, scirc=scirc, nsamples=nsamples, ecirc=ecirc)
end

# TODO add encoding_locs
function naive_pipeline(code::AbstractECC, decoder::AbstractSyndromeDecoder, p_mem; nsamples=10_000, scirc=nothing, ecirc=nothing)
    return naive_pipeline(parity_checks(code), decoder, p_mem, nsamples=nsamples, scirc=scirc, ecirc=ecirc)
end
function naive_pipeline(checks::Stabilizer, decoder::AbstractSyndromeDecoder, p_mem; nsamples=10_000, scirc=nothing, ecirc=nothing)
    if isnothing(scirc)
        scirc, _ = QuantumClifford.ECC.naive_syndrome_circuit(checks)
    end
    O = faults_matrix(checks)

    circuit_Z = Base.copy(scirc)
    circuit_X = Base.copy(scirc)

    s, n = size(checks)
    _, _, r = canonicalize!(Base.copy(checks), ranks=true)
    k = n - r
    pre_X = [sHadamard(i) for i in n-k+1:n]

    md = MixedDestabilizer(checks)
    logview_Z = logicalzview(md)
    logcirc_Z, numLogBits_Z, _ = naive_syndrome_circuit(logview_Z)
    @assert numLogBits_Z == k

    logview_X = logicalxview(md)
    logcirc_X, numLogBits_X, _ = naive_syndrome_circuit(logview_X)
    @assert numLogBits_X == k

    syndrome_bits = 1:s
    logical_bits = s+1:s+k

    x_sub_matrix = O[1:k, :]
    z_sub_matrix = O[k+1:2k, :]

    errors = [PauliError(i,p_mem) for i in 1:n]

    # Z logic circuit
    for gate in logcirc_Z
        type = typeof(gate)
        if type == sMRZ
            push!(circuit_Z, sMRZ(gate.qubit+s, gate.bit+s))
        else
            push!(circuit_Z, type(gate.q1, gate.q2+s))
        end
    end

   # X logic circuit
    for gate in logcirc_X
        type = typeof(gate)
        if type == sMRZ
            push!(circuit_X, sMRZ(gate.qubit+s, gate.bit+s))
        else
            push!(circuit_X, type(gate.q1, gate.q2+s))
        end
    end

    if isnothing(ecirc)
        ecirc_z = fault_tolerant_encoding(circuit_Z) # double syndrome encoding
        ecirc_x = fault_tolerant_encoding(circuit_X) 
    else
        ecirc_z = ecirc
        ecirc_x = ecirc
    end

    full_x_circ = vcat(pre_X, ecirc_x, errors,  circuit_X)
    full_z_circ = vcat(ecirc_z, errors, circuit_Z)

    x_error = evaluate_decoder(decoder, nsamples, full_x_circ, syndrome_bits, logical_bits, x_sub_matrix)
    z_error = evaluate_decoder(decoder, nsamples, full_z_circ, syndrome_bits, logical_bits, z_sub_matrix)

    return x_error, z_error
end

# TODO add encoding_locs
function shor_pipeline(code::AbstractECC, decoder::AbstractSyndromeDecoder, p_mem, nsamples=10_000, cat=nothing, scirc=nothing, ecirc=nothing)
    return shor_pipeline(parity_checks(code), decoder, p_mem, nsamples=nsamples, cat=cat, scirc=scirc, ecirc=ecirc)
end
function shor_pipeline(checks::Stabilizer, decoder::AbstractSyndromeDecoder, p_mem; nsamples=10_000, cat=nothing, scirc=nothing, ecirc=nothing)
    if isnothing(scirc) || isnothing(cat)
        cat, scirc, _ = shor_syndrome_circuit(checks)
    end

    O = faults_matrix(checks)
    circuit_Z = Base.copy(scirc)
    circuit_X = Base.copy(scirc)

    s, n = size(checks)
    _, _, r = canonicalize!(Base.copy(checks), ranks=true)
    k = n - r
    pre_X = [sHadamard(i) for i in n-k+1:n]

    anc_qubits = 0
    for pauli in checks
        anc_qubits += mapreduce(count_ones,+, xview(pauli) .| zview(pauli))
    end
    regbits = anc_qubits + s

    md = MixedDestabilizer(checks)
    logview_Z = logicalzview(md)
    logcirc_Z, _ = naive_syndrome_circuit(logview_Z)

    logview_X = logicalxview(md)
    logcirc_X, _ = naive_syndrome_circuit(logview_X)

    # Z logic circuit
    for gate in logcirc_Z
        type = typeof(gate)
        if type == sMRZ
            push!(circuit_Z, sMRZ(gate.qubit+anc_qubits, gate.bit+regbits))
        else
            push!(circuit_Z, type(gate.q1, gate.q2+anc_qubits))
        end
    end

   # X logic circuit
    for gate in logcirc_X
        type = typeof(gate)
        if type == sMRZ
            push!(circuit_X, sMRZ(gate.qubit+anc_qubits, gate.bit+regbits))
        else
            push!(circuit_X, type(gate.q1, gate.q2+anc_qubits))
        end
    end

    errors = [PauliError(i,p_mem) for i in 1:n]
    if isnothing(ecirc)
        ecirc_z = fault_tolerant_encoding(vcat(cat,circuit_Z)) # double syndrome encoding
        ecirc_x = fault_tolerant_encoding(vcat(cat,circuit_X))
        full_z_circ = vcat(ecirc_z, errors, circuit_Z)
        full_x_circ = vcat(pre_X, ecirc_x, errors, circuit_X)
    else
        full_z_circ = vcat(ecirc, errors, cat, circuit_Z)
        full_x_circ = vcat(pre_X, ecirc, errors, cat, circuit_X)
    end

    syndrome_bits = anc_qubits+1:regbits
    logical_bits = regbits+1:regbits+k

    x_sub_matrix = O[1:k, :]
    z_sub_matrix = O[k+1:2k, :]

    x_error = evaluate_decoder(decoder, nsamples, full_x_circ, syndrome_bits, logical_bits, x_sub_matrix)
    z_error = evaluate_decoder(decoder, nsamples, full_z_circ, syndrome_bits, logical_bits, z_sub_matrix)

    return x_error, z_error
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

    # Special shor compilation
    shor_circuit, shorder = ancil_reindex_pipeline_shor_syndrome(scirc)
    shor_cat = perfect_reindex(cat, shorder)
    shor_post_ec_error_rates_s0 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, shor_cat, shor_circuit, p, 0) for p in error_rates]
    shor_post_ec_error_rates_s10 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, shor_cat, shor_circuit, p, p/10) for p in error_rates]
    shor_post_ec_error_rates_s100 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, shor_cat, shor_circuit, p, p) for p in error_rates]
    shor_x_error_s0 = [shor_post_ec_error_rates_s0[i][1] for i in eachindex(shor_post_ec_error_rates_s0)]
    shor_z_error_s0 = [shor_post_ec_error_rates_s0[i][2] for i in eachindex(shor_post_ec_error_rates_s0)]
    shor_x_error_s10 = [shor_post_ec_error_rates_s10[i][1] for i in eachindex(shor_post_ec_error_rates_s10)]
    shor_z_error_s10 = [shor_post_ec_error_rates_s10[i][2] for i in eachindex(shor_post_ec_error_rates_s10)]
    shor_x_error_s100 = [shor_post_ec_error_rates_s100[i][1] for i in eachindex(shor_post_ec_error_rates_s100)]
    shor_z_error_s100 = [shor_post_ec_error_rates_s100[i][2] for i in eachindex(shor_post_ec_error_rates_s100)]

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

    # Shor Plots
    scatter!(error_rates, shor_x_error_s0, label="Shor compiled circuit with no shift errors", color=:orange, marker=:circle)
    scatter!(error_rates, shor_x_error_s10, label="Shor compiled circuit with shift errors = p/10", color=:orange, marker=:utriangle)
    scatter!(error_rates, shor_x_error_s100, label="Shor compiled circuit with shift errors = p", color=:orange, marker=:star8)

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

    # Shor Plots
    scatter!(error_rates, shor_z_error_s0, label="Shor compiled circuit with no shift errors", color=:orange, marker=:circle)
    scatter!(error_rates, shor_z_error_s10, label="Shor compiled circuit with shift errors = p/10", color=:orange, marker=:utriangle)
    scatter!(error_rates, shor_z_error_s100, label="Shor compiled circuit with shift errors = p", color=:orange, marker=:star8)
    
    f_z[1,2] = Legend(f_z, ax, "Error Rates")
    return f_x, f_z
end

"""Same as [`vary_shift_errors_plot_shor_syndrome`](@ref) but also applies realistic noise pararmeters related to:
- Error rate per shift
- Error rate due to waiting (decoherence)
- Two qubit gate fidelity
"""
function realistic_noise_logical_physical_error(code::AbstractECC, p_shift=0.0001, p_wait=1-exp(-14.5/28_000), p_gate=1-0.998; name=string(typeof(code)))
    title = name*" Circuit - Shor Syndrome Circuit"
    checks = parity_checks(code)
    cat, scirc, anc_qubits, bit_indices = shor_syndrome_circuit(checks)
    ecirc = naive_encoding_circuit(code)
    nframes = 20_000
    m = 1/2
    p_gate= 0

    error_rates = 0.000:0.00150:0.30
    # Uncompiled errors
    post_ec_error_rates_s0 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, cat, add_two_qubit_gate_noise(scirc, p_gate), p, p_shift, p_wait, nframes=nframes) for p in error_rates]
    post_ec_error_rates_s1 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, cat, add_two_qubit_gate_noise(scirc, p_gate*m), p, p_shift*m, 1-exp(-14.5*m/28_000), nframes=nframes) for p in error_rates]
    x_error_s0 = [post_ec_error_rates_s0[i][1] for i in eachindex(post_ec_error_rates_s0)]
    z_error_s0 = [post_ec_error_rates_s0[i][2] for i in eachindex(post_ec_error_rates_s0)]
    x_error_s1 = [post_ec_error_rates_s1[i][1] for i in eachindex(post_ec_error_rates_s1)]
    z_error_s1 = [post_ec_error_rates_s1[i][2] for i in eachindex(post_ec_error_rates_s1)]

    # # Anc Compiled circuit
    # new_circuit, order = ancil_reindex_pipeline(scirc)
    # new_cat = perfect_reindex(cat, order)
    # compiled_post_ec_error_rates_s0 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, new_cat, add_two_qubit_gate_noise(new_circuit, p_gate), p, p_shift, p_wait, nframes=nframes) for p in error_rates]
    # compiled_post_ec_error_rates_s1 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, new_cat, add_two_qubit_gate_noise(new_circuit, p_gate*m), p, p_shift*m, 1-exp(-14.5*m/28_000), nframes=nframes) for p in error_rates]
    # compiled_x_error_s0 = [compiled_post_ec_error_rates_s0[i][1] for i in eachindex(compiled_post_ec_error_rates_s0)]
    # compiled_z_error_s0 = [compiled_post_ec_error_rates_s0[i][2] for i in eachindex(compiled_post_ec_error_rates_s0)]
    # compiled_x_error_s1 = [compiled_post_ec_error_rates_s1[i][1] for i in eachindex(compiled_post_ec_error_rates_s1)]
    # compiled_z_error_s1 = [compiled_post_ec_error_rates_s1[i][2] for i in eachindex(compiled_post_ec_error_rates_s1)]

    # Special shor syndrome Compiled circuit
    shor_new_circuit, shor_order = ancil_reindex_pipeline_shor_syndrome(scirc)
    shor_cat = perfect_reindex(cat, shor_order)
    shor_post_ec_error_rates_s0 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, shor_cat, add_two_qubit_gate_noise(shor_new_circuit, p_gate), p, p_shift, p_wait, nframes=nframes) for p in error_rates]
    shor_post_ec_error_rates_s1 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, shor_cat, add_two_qubit_gate_noise(shor_new_circuit, p_gate*m), p, p_shift*m, 1-exp(-14.5*m/28_000), nframes=nframes) for p in error_rates]
    shor_x_error_s0 = [shor_post_ec_error_rates_s0[i][1] for i in eachindex(shor_post_ec_error_rates_s0)]
    shor_z_error_s0 = [shor_post_ec_error_rates_s0[i][2] for i in eachindex(shor_post_ec_error_rates_s0)]
    shor_x_error_s1 = [shor_post_ec_error_rates_s1[i][1] for i in eachindex(shor_post_ec_error_rates_s1)]
    shor_z_error_s1 = [shor_post_ec_error_rates_s1[i][2] for i in eachindex(shor_post_ec_error_rates_s1)]

    f_x = Figure(resolution=(1100,900))
    ax = f_x[1,1] = Axis(f_x, xlabel="single (qu)bit error rate - AKA error after encoding",ylabel="Logical error rate",title=title*" Logical X")
    lim = max(error_rates[end])
    lines!([0,lim], [0,lim], label="single bit", color=:black)

    # Uncompiled Plots
    scatter!(error_rates, x_error_s0, label="Uncompiled with realistic errors", color=:red, marker=:star8)
    scatter!(error_rates, x_error_s1, label="Uncompiled w/ moderately optimistic (m=1/2) params", color=:red, marker=:utriangle)

    # # Compiled Plots
    # scatter!(error_rates, compiled_x_error_s0, label="Compiled (AC + GS) with realistic errors", color=:blue, marker=:star8)
    # scatter!(error_rates, compiled_x_error_s1, label="Compiled w/ moderately optimistic (m=1/3) params", color=:blue, marker=:utriangle)

    # Shor syndrome specialized
    scatter!(error_rates, shor_x_error_s0, label="Shor synd specialized with realistic errors", color=:green, marker=:star8)
    scatter!(error_rates, shor_x_error_s1, label="Shor synd specialized w/ moderately optimistic (m=1/2) params", color=:green, marker=:utriangle)

    f_x[1,2] = Legend(f_x, ax, "Error Rates")

    f_z = Figure(resolution=(1100,900))
    ax = f_z[1,1] = Axis(f_z, xlabel="single (qu)bit error rate - AKA error after encoding",ylabel="Logical error rate",title=title*" Logical Z")
    lim = max(error_rates[end])
    lines!([0,lim], [0,lim], label="single bit", color=:black)

    # Uncompiled Plots
    scatter!(error_rates, z_error_s0, label="Uncompiled with realistic errors", color=:red, marker=:star8)
    scatter!(error_rates, z_error_s1, label="Uncompiled w/ moderately optimistic (m=1/2) params", color=:red, marker=:utriangle)

    # # Compiled Plots
    # scatter!(error_rates, compiled_z_error_s0, label="Compiled (AC + GS) with realistic errors", color=:blue, marker=:star8)
    # scatter!(error_rates, compiled_z_error_s1, label="Compiled w/ moderately optimistic (m=1/3) params", color=:blue, marker=:utriangle)

    # Shor syndrome specialized
    scatter!(error_rates, shor_z_error_s0, label="Shor synd specialized with realistic errors", color=:green, marker=:star8)
    scatter!(error_rates, shor_z_error_s1, label="Shor synd specialized w/ moderately optimistic (m=1/2) params", color=:green, marker=:utriangle)

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

# TODO all the below stuff is outdated and needs to be delelted or reworked
# """Wrapper for QuantumClifford.ECC.CSS_naive_error_correction_pipeline()
# """
# function evaluate_code_decoder_FTecirc_pf_krishna(code::old_CSS, scirc, p_init, p_shift=0 ; nframes=1_000, max_iters = 25)
#     s, n = size(code.tableau)
#     _, _, r = canonicalize!(Base.copy(code.tableau), ranks=true)
#     k = n - r

#     if p_shift != 0
#         non_mz, mz = clifford_grouper(scirc)
#         non_mz = calculate_shifts(non_mz)
#         scirc = []

#         first_shift = true
#         for subcircuit in non_mz
#             # Shift!
#             if !first_shift
#                 append!(scirc, [PauliError(i,p_shift) for i in n+1:n+s])
#             end
#             append!(circuit_Z, subcircuit)
#             first_shift = false
#         end
#         append!(scirc, mz)
#     end

#     return QuantumClifford.ECC.CSS_naive_error_correction_pipeline(code, p_init, nframes=nframes, scirc=scirc, max_iters=max_iters) 
# end

# """Wrapper for CSS_shor_error_correction_pipeline()
# """
# function evaluate_code_FTencode_FTsynd_Krishna(code::old_CSS, cat, scirc, p_init, p_shift=0, p_wait=0; nframes=10_000, max_iters = 25)
#     s, n = size(code.tableau)

#     anc_qubits = 0
#     for pauli in code.tableau
#         anc_qubits += mapreduce(count_ones,+, xview(pauli) .| zview(pauli))
#     end

#     if p_shift != 0
#         non_mz, mz = clifford_grouper(scirc)
#         non_mz = calculate_shifts(non_mz)
#         scirc = []

#         current_delta = 0 # Starts with ancil qubits not lined up with any of the physical ones
#         for subcircuit in non_mz
#             new_delta = abs(subcircuit[1].q2-subcircuit[1].q1)
#             hops_to_new_delta = abs(new_delta-current_delta)
#             shift_error = p_shift * hops_to_new_delta

#             # Shift!
#             append!(scirc, [PauliError(i,shift_error) for i in 1:n])
            
#             # TODO Should this be random Pauli error or just Z error?
#             append!(scirc, [PauliError(i,p_wait) for i in n+1:n+anc_qubits])

#             append!(scirc, subcircuit)
#         end
#         append!(scirc, mz)
#     end
#     return QuantumClifford.ECC.CSS_shor_error_correction_pipeline(code, p_init, cat=cat, scirc=scirc, nframes=nframes, max_iters=max_iters)
# end

# # This had terrible results - The large LDPC codes need to the gate fidelity to be about a factor of 10^4 
# """Effects of compilation against varying memory error after encoding + other realistic sources of error""" 
# function realistic_noise_logical_physical_error_ldpc(code::old_CSS, p_shift=0.0001, p_wait=1-exp(-14.5/28_000), p_gate=1-0.998; name="")
#     title = name*" Circuit - Shor Syndrome Circuit"

#     checks = code.tableau
#     cat, scirc, _ = shor_syndrome_circuit(checks)

#     nframes = 1_000
#     m = 1/10
#     p_wait = 0 # this term is fine
#     #p_gate = 0 # gate error alone causes terrible performance
#     p_shift = 0

#     error_rates = exp10.(range(-5,-1,length=40))
#     # Uncompiled errors
#     # noisy_scirc = add_two_qubit_gate_noise(scirc, p_gate)
#     # post_ec_error_rates_s0 = [evaluate_code_FTencode_FTsynd_Krishna(code, cat, noisy_scirc, p, p_shift, p_wait, nframes=nframes) for p in error_rates]
#     noisy_scirc_m = add_two_qubit_gate_noise(scirc, p_gate*m)
#     post_ec_error_rates_s1 = [evaluate_code_FTencode_FTsynd_Krishna(code, cat, noisy_scirc_m, p, p_shift*m, 1-exp(-14.5*m/28_000), nframes=nframes) for p in error_rates]
#     # x_error_s0 = [post_ec_error_rates_s0[i][1] for i in eachindex(post_ec_error_rates_s0)]
#     # z_error_s0 = [post_ec_error_rates_s0[i][2] for i in eachindex(post_ec_error_rates_s0)]
#     x_error_s1 = [post_ec_error_rates_s1[i][1] for i in eachindex(post_ec_error_rates_s1)]
#     z_error_s1 = [post_ec_error_rates_s1[i][2] for i in eachindex(post_ec_error_rates_s1)]

#     # # Anc Compiled circuit
#     # new_circuit, order = ancil_reindex_pipeline(scirc)
#     # new_cat = perfect_reindex(cat, order)
#     # noise_new_circuit = add_two_qubit_gate_noise(new_circuit, p_gate)
#     # compiled_post_ec_error_rates_s0 = [evaluate_code_FTencode_FTsynd_Krishna(code, new_cat, noise_new_circuit, p, p_shift, p_wait, nframes=nframes) for p in error_rates]
#     # noise_new_circuit_m = add_two_qubit_gate_noise(new_circuit, p_gate*m)
#     # compiled_post_ec_error_rates_s1 = [evaluate_code_FTencode_FTsynd_Krishna(code, new_cat, noise_new_circuit_m, p, p_shift*m, 1-exp(-14.5*m/28_000), nframes=nframes) for p in error_rates]
#     # compiled_x_error_s0 = [compiled_post_ec_error_rates_s0[i][1] for i in eachindex(compiled_post_ec_error_rates_s0)]
#     # compiled_z_error_s0 = [compiled_post_ec_error_rates_s0[i][2] for i in eachindex(compiled_post_ec_error_rates_s0)]
#     # compiled_x_error_s1 = [compiled_post_ec_error_rates_s1[i][1] for i in eachindex(compiled_post_ec_error_rates_s1)]
#     # compiled_z_error_s1 = [compiled_post_ec_error_rates_s1[i][2] for i in eachindex(compiled_post_ec_error_rates_s1)]

#     # Special shor syndrome Compiled circuit
#     new_circuit, order = ancil_reindex_pipeline_shor_syndrome(scirc)
#     new_cat = perfect_reindex(cat, order)
#     # noise_new_circuit = add_two_qubit_gate_noise(new_circuit, p_gate)
#     # shor_post_ec_error_rates_s0 = [evaluate_code_FTencode_FTsynd_Krishna(code, new_cat, noise_new_circuit, p, p_shift, p_wait, nframes=nframes) for p in error_rates]
#     noise_new_circuit_m = add_two_qubit_gate_noise(new_circuit, p_gate*m)
#     shor_post_ec_error_rates_s1 = [evaluate_code_FTencode_FTsynd_Krishna(code, new_cat, noise_new_circuit_m, p, p_shift*m, 1-exp(-14.5*m/28_000), nframes=nframes) for p in error_rates]
#     # shor_x_error_s0 = [shor_post_ec_error_rates_s0[i][1] for i in eachindex(shor_post_ec_error_rates_s0)]
#     # shor_z_error_s0 = [shor_post_ec_error_rates_s0[i][2] for i in eachindex(shor_post_ec_error_rates_s0)]
#     shor_x_error_s1 = [shor_post_ec_error_rates_s1[i][1] for i in eachindex(shor_post_ec_error_rates_s1)]
#     shor_z_error_s1 = [shor_post_ec_error_rates_s1[i][2] for i in eachindex(shor_post_ec_error_rates_s1)]

#     f_x = Figure(resolution=(1100,900))
#     ax = f_x[1,1] = Axis(f_x, xlabel="Log10 single (qu)bit error rate - AKA error after encoding",ylabel="Log10 Logical error rate",title=title*" Logical X")

#     lines!([-5, 0], [-5, 0], label="single bit", color=:black)

#     # Uncompiled Plots
#     #scatter!(log10.(error_rates), log10.(x_error_s0), label="Uncompiled with realistic errors", color=:red, marker=:star8)
#     scatter!(log10.(error_rates), log10.(x_error_s1), label="Uncompiled w/ moderately optimistic (m=1/10) params", color=:red, marker=:utriangle)

#     # # Compiled Plots
#     # scatter!(log10.(error_rates), log10.(compiled_x_error_s0), label="Compiled (AC + GS) with realistic errors", color=:blue, marker=:star8)
#     # scatter!(log10.(error_rates), log10.(compiled_x_error_s1), label="Compiled w/ moderately optimistic (m=1/10) params", color=:blue, marker=:utriangle)

#     # Shor specialized compiled Plots
#     #scatter!(log10.(error_rates), log10.(shor_x_error_s0), label="Compiled (AC + GS) with realistic errors", color=:green, marker=:star8)
#     scatter!(log10.(error_rates), log10.(shor_x_error_s1), label="Compiled w/ moderately optimistic (m=1/10) params", color=:green, marker=:utriangle)

#     f_x[1,2] = Legend(f_x, ax, "Error Rates")

#     f_z = Figure(resolution=(1100,900))
#     ax = f_z[1,1] = Axis(f_z, xlabel="Log10 single (qu)bit error rate - AKA error after encoding",ylabel="Log10 Logical error rate",title=title*" Logical Z")

#     lines!([-5, 0], [-5, 0], label="single bit", color=:black)

#     # Uncompiled Plots
#     #scatter!(log10.(error_rates), log10.(z_error_s0), label="Uncompiled with realistic errors", color=:red, marker=:star8)
#     scatter!(log10.(error_rates), log10.(z_error_s1), label="Uncompiled w/ moderately optimistic (m=1/10) params", color=:red, marker=:utriangle)

#     # # Compiled Plots
#     # scatter!(log10.(error_rates), log10.(compiled_z_error_s0), label="Compiled (AC + GS) with realistic errors", color=:blue, marker=:star8)
#     # scatter!(log10.(error_rates), log10.(compiled_z_error_s1), label="Compiled w/ moderately optimistic (m=1/10) params", color=:blue, marker=:utriangle)

#     # Shor specialized compiled Plots
#     #scatter!(log10.(error_rates), log10.(shor_z_error_s0), label="Compiled (AC + GS) with realistic errors", color=:green, marker=:star8)
#     scatter!(log10.(error_rates), log10.(shor_z_error_s1), label="Compiled w/ moderately optimistic (m=1/10) params", color=:green, marker=:utriangle)
#     f_z[1,2] = Legend(f_z, ax, "Error Rates")
#     return f_x, f_z
# end
